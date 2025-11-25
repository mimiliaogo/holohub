"""
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
SPDX-License-Identifier: Apache-2.0

VoxelStreamToVolume operator: converts streaming voxel data to dense 3D volume.
"""

import cupy as cp
import numpy as np
from holoscan.core import Operator, OperatorSpec, ConditionType
from holoscan.gxf import Entity
import torch
import nibabel as nib  # Only needed for debug NIfTI saving
from nilearn import image  # Not currently used

class VoxelStreamToVolumeOp(Operator):
    """
    Convert streaming dense voxel data [I, J, K, 2] into a 3D volume tensor for VolumeRendererOp.

    Inputs:
    - affine_4x4: np.ndarray shape (4, 4) (processed once if provided)
    - hb_voxel_data: np.ndarray shape (I, J, K, n_channels) where last dim is channels [HbO, HbR]

    Outputs:
    - volume: holoscan.gxf.Entity containing a tensor named "volume" with shape (Z,Y,X)
    - spacing: np.ndarray shape (3,) derived from affine
    - permute_axis: np.ndarray shape (3,) derived from affine
    - flip_axes: np.ndarray shape (3,) derived from affine
    """

    def __init__(self, fragment, *args, **kwargs):
        self.selected_channel = kwargs.pop("selected_channel", 0)
        self.label_path = kwargs.pop("label_path", "/workspace/holohub/data/bci_visualization/resampled_labels.npz")
        self.roi_labels = kwargs.pop("roi_labels", [3, 4])  # List of label values to consider as ROI
        
        super().__init__(fragment, *args, **kwargs)
        
        # Internal state
        self.affine = None
        
        # Metadata, set from the first frame, reused for subsequent frames
        self.dims = None  # np.array([X, Y, Z], dtype=np.uint32)
        self.out_spacing = None  # np.ndarray float32 (3,)
        self.permute_axis = None  # np.ndarray uint32 (3,)
        self.flip_axes = None  # np.ndarray bool (3,)
        self.min_value = 0.0  # float
        self.max_value = 0.002  # Clip value to [0, 0.002], hardcoded for now
        self.roi_mask = None  # np.ndarray bool (I, J, K)

        # Labels for brain anatomy
        self.labels = None  # np.ndarray (I, J, K)
        
    def setup(self, spec: OperatorSpec):
        spec.input("affine_4x4").condition(ConditionType.NONE)  # (4, 4), only emit at the first frame
        spec.input("hb_voxel_data")  # (I, J, K, n_channels)
        
        spec.output("volume")
        spec.output("spacing")
        spec.output("permute_axis")
        spec.output("flip_axes")
        # spec.output("extent")  # Not used by downstream operators
    
    def compute(self, op_input, op_output, context):
        
        # Receive Hb voxel data
        hb_voxel = op_input.receive("hb_voxel_data") # IJK space, every frame
        
        # Check voxel data is valid
        if hb_voxel is None:
            raise ValueError("VoxelStreamToVolume: No voxel data received this frame")
        hb_voxel = np.asarray(hb_voxel)
        if hb_voxel.ndim != 4 or hb_voxel.shape[-1] < 1:
            raise ValueError(f"VoxelStreamToVolume: Invalid voxel data shape: {hb_voxel.shape}, expected 4D with channels")
        
        # Receive affine matrix only at the first frame
        affine = op_input.receive("affine_4x4")
        if affine is not None:
            self.affine = np.array(affine, dtype=np.float32).reshape(4, 4)
            # Derive spacing/orientation from affine
            self.out_spacing, self.permute_axis, self.flip_axes = self._derive_orientation_from_affine(self.affine)
            # Set metadata from the first frame
            self.dims = hb_voxel.shape[:3] # (I, J, K)
            # self.min_value = np.min(np.abs(hb_voxel)[..., self.selected_channel])
            # self.max_value = np.max(np.abs(hb_voxel)[..., self.selected_channel])
            print("VoxelStreamToVolume: Received and processed affine matrix")
        
        # Check if affine has been set at least once
        if self.affine is None:
            raise ValueError("VoxelStreamToVolume: No affine matrix received")

        # Load and resize labels only at the first frame
        if self.labels is None and self.label_path:
            self.labels = np.load(self.label_path)['data']
            print(f'VoxelStreamToVolume: Loaded labels from {self.label_path}: {self.labels.shape}')
            # self.labels = self._resize_labels_to_volume_dims(self.labels, self.dims)
            # print('VoxelStreamToVolume: Reshaped labels to: ', self.labels.shape)
            
            # Compute ROI mask once from specified label values
            if len(self.roi_labels) > 0:
                self.roi_mask = np.isin(self.labels, self.roi_labels)
                print(f'VoxelStreamToVolume: Computed ROI mask for labels {self.roi_labels}')
            else:
                print('VoxelStreamToVolume: No ROI labels specified, all voxels will be processed')

        # TODO(MImi): why z, y, x?
        # Select channel and arrange as (Z, Y, X)
        vol_xyz = hb_voxel[..., self.selected_channel].astype(np.float32, copy=False)  # (X, Y, Z)

        # Debug prints (commented out for performance)
        # print('VoxelStreamToVolume: before normalize: ', np.min(vol_xyz), np.max(vol_xyz))
        
        # FIXME(Mimi): separate out positive and negative values
        # Ignore negative values by taking absolute value
        vol_xyz = np.abs(vol_xyz)
        
        # Normalize to [-1024, 3071] and process activated voxels
        vol_xyz = self._normalize_and_process_activated_voxels(vol_xyz)
        
        # print('VoxelStreamToVolume: after normalize: ', np.min(vol_xyz), np.max(vol_xyz))

        # FIXME(Mimi): debug - save vol_xyz to nifti (commented out for performance)
        img = nib.Nifti1Image(vol_xyz, self.affine)
        nib.save(img, '/workspace/holohub/render_frontend/datasets/pipeline_dataset/volume.nii.gz')

        # Transpose to (Z, Y, X) 
        vol_zyx = np.transpose(vol_xyz, (2, 1, 0)).copy() # FIXME(Mimi): w/o copy, CUDA error
        
        # Emit outputs
        op_output.emit({"volume": cp.asarray(vol_zyx)}, "volume")
        op_output.emit(self.out_spacing, "spacing", "std::array<float, 3>")
        op_output.emit(self.permute_axis, "permute_axis", "std::array<uint32_t, 3>")
        op_output.emit(self.flip_axes, "flip_axes", "std::array<bool, 3>")

    def _resize_labels_to_volume_dims(self, labels: np.ndarray, dims: tuple):
        """
        Resize labels to the same dimensions as the voxel data.
        """
        # Convert labels to torch tensor
        labels_tensor = torch.from_numpy(labels).float()
        
        # Add batch and channel dimensions for interpolate: (N, C, D, H, W)
        if labels_tensor.ndim == 3:
            labels_tensor = labels_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
        else:
            raise ValueError(f"Expected 3D labels, got shape {labels_tensor.shape}")
        
        # Resize labels to the same dimensions as the voxel data
        labels_tensor = torch.nn.functional.interpolate(labels_tensor, size=dims, mode='nearest')

        # Convert to int32
        labels_tensor = labels_tensor.int()
        
        # Remove batch and channel dimensions and convert back to numpy array
        return labels_tensor.squeeze(0).squeeze(0).numpy().astype(np.int32)

    def _derive_orientation_from_affine(self, affine_4x4: np.ndarray):
        """
        Derive spacing, axis permutation, and flips from affine.
        spacing: voxel sizes along data axes (I,J,K) mapped to [X,Y,Z] ordering
        permute_axis: for each data axis (I,J,K), index of world axis (X=0,Y=1,Z=2)
        flip_axes: whether the axis is flipped (negative orientation)
        """
        # TODO(Mimi): check if this is correct
        R = affine_4x4[:3, :3].astype(np.float32)
        # spacing along data axes (length of each column)
        spacing_ijk = np.linalg.norm(R, axis=0).astype(np.float32)
        # Avoid zeros
        spacing_ijk[spacing_ijk == 0] = 1.0
        # Direction cosines
        dirs = R / spacing_ijk
        permute = np.zeros(3, dtype=np.uint32)
        flips = np.zeros(3, dtype=bool)
        for a in range(3):  # data axis I,J,K
            world_axis = int(np.argmax(np.abs(dirs[:, a])))
            permute[a] = world_axis
            flips[a] = dirs[world_axis, a] < 0
        # spacing returned in [X, Y, Z] order by mapping data spacings
        spacing_xyz = np.zeros(3, dtype=np.float32)
        for a in range(3):
            spacing_xyz[permute[a]] = spacing_ijk[a]


        # FIXME(Mimi): hardcoded spacing, permute, flip
        # spacing_xyz = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        permute = np.array([0, 2, 1], dtype=np.uint32)
        flips = np.array([True, False, False], dtype=bool)
        return spacing_xyz.astype(np.float32), permute.astype(np.uint32), flips.astype(bool)

    def _normalize_and_process_activated_voxels(self, vol_zyx: np.ndarray, normalize_min_value: float = -1024, normalize_max_value: float = 3071):
        """
        Normalize the volume to [min_value, max_value] and process activated voxels.    
        Returns:
            vol_zyx: np.ndarray shape (Z, Y, X) with values in [0, 1]
        """
        # Every voxel has a different min and max value
        # We simply use the first frame's min and max value for normalization
        # FIXME(Mimi): should normalize in the upstream operator
        
        # Clip values to [self.min_value, self.max_value]
        vol_zyx = np.clip(vol_zyx, self.min_value, self.max_value)
        
        # Normalize to normalize_min_value and normalize_max_value
        vol_zyx = (vol_zyx - self.min_value) / (self.max_value - self.min_value) * (normalize_max_value - normalize_min_value) + normalize_min_value
        
        # Apply ROI mask: set non-ROI voxels to max_value (mask is computed once at initialization)
        if self.roi_mask is not None:
            vol_zyx = np.where(~self.roi_mask, normalize_max_value, vol_zyx)
        
        # Debug code (commented out for performance)
        # print('VoxelStreamToVolume: min_value: ', self.min_value, 'max_value: ', self.max_value, 'np.min(vol_zyx): ', np.min(vol_zyx), 'np.max(vol_zyx): ', np.max(vol_zyx))
        # vol_zyx_normalized = (vol_zyx - normalize_min_value) / (normalize_max_value - normalize_min_value)
        # import matplotlib.pyplot as plt
        # plt.hist(vol_zyx_normalized.flatten(), bins=100, log=True)
        # plt.savefig('vol_zyx_distribution_normalized.png')
        
        return vol_zyx