"""
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
SPDX-License-Identifier: Apache-2.0

Simple voxel loader that generates synthetic voxel data for testing.
"""

import numpy as np
from holoscan.core import Operator, OperatorSpec
import os
import nibabel as nib
import holoscan as hs
from nilearn import image
import time
from nilearn.image import resample_to_img


class SimpleVoxelLoaderOp(Operator):
    """
    Load a hardcoded NIfTI file and stream voxel data per frame.

    Outputs:
    - affine_4x4: np.ndarray shape (4, 4) from NIfTI header (emitted once)
    - hb_voxel_data: np.ndarray shape (I, J, K, 1) with channel [HbO] (emitted every frame)

    """

    def __init__(self, fragment, *args, **kwargs):
        self.nifti_path = kwargs.pop("nifti_path", "/workspace/holohub/data/bci_visualization/study-trustreliabilitystudy_sub-trust1023s175_desc-563862f_HbO.nii.gz")
        
        super().__init__(fragment, *args, **kwargs)
        
        # Internal state
        self.sent_affine = False
        self.frame_count = 0
        self.affine_4x4 = None
        self.volume_4d = None  # (I, J, K, T)
        self.nifti_loaded = False

    def setup(self, spec: OperatorSpec):
        spec.output("affine_4x4")
        spec.output("hb_voxel_data")
    
    def start(self):
        self._load_nifti()

    def compute(self, op_input, op_output, context):
        # Check if we have volume data
        if self.volume_4d is None:
            raise RuntimeError("SimpleVoxelLoader: No volume data loaded")

        # Emit affine once
        if not self.sent_affine and self.affine_4x4 is not None:
            op_output.emit(self.affine_4x4, "affine_4x4")
            self.sent_affine = True
            print("SimpleVoxelLoader: Sent affine matrix")

        # Wrap frame count to loop through available frames
        num_frames = self.volume_4d.shape[3]
        current_frame = self.frame_count % num_frames

        # Emit voxel data every frame
        hb_voxel_per_frame = self.volume_4d[..., current_frame]
        
        print(f'SimpleVoxelLoader: Frame {current_frame}, min/max: {np.min(hb_voxel_per_frame):.4f}, {np.max(hb_voxel_per_frame):.4f}')
        hb_voxel_per_frame = hb_voxel_per_frame[..., np.newaxis]
        
        # Run in 10hz
        time.sleep(1)
        op_output.emit(hb_voxel_per_frame, "hb_voxel_data") # [I, J, K, 1]

        self.frame_count += 1

    def _load_nifti(self):
        """
        Load the configured NIfTI file. If missing or nibabel unavailable,
        raise an error.
        """
        if not self.nifti_path:
            raise ValueError("SimpleVoxelLoader: No NIfTI file path provided")
            
        if not os.path.exists(self.nifti_path):
            raise FileNotFoundError(f"SimpleVoxelLoader: NIfTI file not found: {self.nifti_path}")
            
        if nib is None:
            raise ImportError("SimpleVoxelLoader: nibabel is required to load NIfTI files")
            
        try:
            img = nib.load(self.nifti_path)
            volume_4d = img.get_fdata()
            affine = img.affine

            self.affine_4x4 = np.asarray(affine, dtype=np.float32)
            self.volume_4d = np.asarray(volume_4d, dtype=np.float32)
            self.volume_dims = self.volume_4d.shape[:3] # (I, J, K)
            print(f'SimpleVoxelLoader: Loaded NIfTI from {self.nifti_path}')
            print(f'SimpleVoxelLoader: Volume shape: {self.volume_4d.shape}')
            return
        except Exception as e:
            raise RuntimeError(f"SimpleVoxelLoader: Failed to load NIfTI '{self.nifti_path}': {e}")
