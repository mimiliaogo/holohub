"""
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
SPDX-License-Identifier: Apache-2.0

BCI Visualization Application - streams synthetic voxel data and renders as 3D volume.
"""

import os
import argparse


from holoscan.conditions import CountCondition
from holoscan.core import Application
from holoscan.operators import HolovizOp
from holoscan.resources import CudaStreamPool, UnboundedAllocator

# Import local operators
from simple_voxel_loader import SimpleVoxelLoaderOp
from voxel_stream_to_volume import VoxelStreamToVolumeOp

from holohub.volume_loader import VolumeLoaderOp
from holohub.volume_renderer import VolumeRendererOp


class BciVisualizationApp(Application):
    """BCI Visualization Application with ClaraViz."""

    def __init__(self, 
        argv=None,
        *args,
        render_config_file,
        density_min,
        density_max,
        label_path=None,
        roi_labels=None,
        nifti_path=None,
        **kwargs,
    ):
        self._rendering_config = render_config_file
        self._density_min = density_min
        self._density_max = density_max
        self._label_path = label_path
        self._roi_labels = roi_labels
        self._nifti_path = nifti_path

        super().__init__(argv, *args, **kwargs)

    def compose(self):
        volume_allocator = UnboundedAllocator(self, name="allocator")
        cuda_stream_pool = CudaStreamPool(
            self,
            name="cuda_stream",
            dev_id=0,
            stream_flags=0,
            stream_priority=0,
            reserved_size=1,
            max_size=5,
        )

        # Selection
        selected_channel = 0  # 0: HbO, 1: HbR

        # Resources
        volume_allocator = UnboundedAllocator(self, name="allocator")
        cuda_stream_pool = CudaStreamPool(
            self,
            name="cuda_stream",
            dev_id=0,
            stream_flags=0,
            stream_priority=0,
            reserved_size=1,
            max_size=5,
        )

        # Operators
        simple_voxel_loader_args = {}
        if self._nifti_path:
            simple_voxel_loader_args["nifti_path"] = self._nifti_path
            
        simple_voxel_loader = SimpleVoxelLoaderOp(
            self,
            # CountCondition(self, count=1),
            name="simple_voxel_loader",
            **simple_voxel_loader_args,
        )

        voxel_to_volume_args = {
            "selected_channel": selected_channel,
        }
        if self._label_path:
            voxel_to_volume_args["label_path"] = self._label_path
        if self._roi_labels:
            voxel_to_volume_args["roi_labels"] = self._roi_labels
            
        voxel_to_volume = VoxelStreamToVolumeOp(
            self,
            name="voxel_to_volume",
            **voxel_to_volume_args,
        )

        volume_renderer_args = {}
        if self._density_min:
            volume_renderer_args["density_min"] = self._density_min
        if self._density_max:
            volume_renderer_args["density_max"] = self._density_max

        volume_renderer = VolumeRendererOp(
            self,
            name="volume_renderer",
            config_file=self._rendering_config,
            allocator=volume_allocator,
            alloc_width=1024, # TODO(Mimi): check what is this for
            alloc_height=768,
            cuda_stream_pool=cuda_stream_pool,
            **volume_renderer_args,
        )

        holoviz = HolovizOp(
            self,
            name="holoviz",
            window_title="BCI Visualization with ClaraViz",
            enable_camera_pose_output=True,
            cuda_stream_pool=cuda_stream_pool,
        )

        # Connect operators
        # simple_voxel_loader → voxel_to_volume
        self.add_flow(simple_voxel_loader, voxel_to_volume, {
            ("affine_4x4", "affine_4x4"),
            ("hb_voxel_data", "hb_voxel_data"),
        })

        # voxel_to_volume → volume_renderer
        self.add_flow(voxel_to_volume, volume_renderer, {
            ("volume", "density_volume"),
            ("spacing", "density_spacing"),
            ("permute_axis", "density_permute_axis"),
            ("flip_axes", "density_flip_axes"),
        })

        # volume_renderer ↔ holoviz
        self.add_flow(volume_renderer, holoviz, {("color_buffer_out", "receivers")})
        self.add_flow(holoviz, volume_renderer, {("camera_pose_output", "camera_pose")})


def main():
    
    
    parser = argparse.ArgumentParser(description="BCI Visualization Application", add_help=False)
    parser.add_argument(
        "-c",
        "--config",
        action="store",
        dest="config",
        help="Name of the renderer JSON configuration file to load",
    )

    parser.add_argument(
        "-i",
        "--density_min",
        action="store",
        type=int,
        dest="density_min",
        help="Set the minimum of the density element values. If not set this is calculated from the"
        "volume data. In practice CT volumes have a minimum value of -1024 which corresponds to"
        "the lower value of the Hounsfield scale range usually used.",
    )
    parser.add_argument(
        "-a",
        "--density_max",
        action="store",
        type=int,
        dest="density_max",
        help="Set the maximum of the density element values. If not set this is calculated from the"
        "volume data. In practice CT volumes have a maximum value of 3071 which corresponds to"
        "the upper value of the Hounsfield scale range usually used.",
    )
    
    parser.add_argument(
        "-l",
        "--label_path",
        action="store",
        type=str,
        dest="label_path",
        help="Path to the NPZ file containing brain anatomy labels. If not provided, uses default path.",
    )
    
    parser.add_argument(
        "-r",
        "--roi_labels",
        action="store",
        type=str,
        dest="roi_labels",
        help="Comma-separated list of label values to use as ROI (e.g., '3,4'). Default is '3,4'.",
    )
    
    parser.add_argument(
        "-n",
        "--nifti_path",
        action="store",
        type=str,
        dest="nifti_path",
        help="Path to the NIfTI file containing 4D volume data (I, J, K, T). If not provided, uses default path.",
    )

    parser.add_argument(
        "-h", "--help", action="help", default=argparse.SUPPRESS, help="Help message"
    )
    
    args = parser.parse_args()
    
    # Parse roi_labels from comma-separated string to list of integers
    roi_labels = None
    if args.roi_labels:
        try:
            roi_labels = [int(label.strip()) for label in args.roi_labels.split(',')]
        except ValueError:
            print(f"Warning: Invalid roi_labels format '{args.roi_labels}'. Expected comma-separated integers.")
            roi_labels = None

    app = BciVisualizationApp(
        render_config_file=args.config,
        density_min=args.density_min,
        density_max=args.density_max,
        label_path=args.label_path,
        roi_labels=roi_labels,
        nifti_path=args.nifti_path,
    )

    app.run()
    
    print("BCI Visualization Application has finished running.")


if __name__ == "__main__":
    main()

