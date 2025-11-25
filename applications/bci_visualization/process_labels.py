# Load labels.npz and resampled to the size of volume.nii.gz
# from this import d
import numpy as np
import nibabel as nib
import os
import torch

parent_dir = "/workspace/holohub/data/bci_visualization"
nifti_file = "study-trustreliabilitystudy_sub-trust1023s175_desc-563862f_HbO.nii.gz"
labels_path = os.path.join(parent_dir, 'labels.npz')
resampled_activation_volume_4d_path = os.path.join(parent_dir, nifti_file)

labels = np.load(labels_path)['data']
print('nifti path: ', resampled_activation_volume_4d_path)
volume = nib.load(resampled_activation_volume_4d_path)

volume_data = volume.get_fdata()
dims = volume_data.shape[:3]
print('Volume data dims: ', dims)


# Convert labels to torch tensor
labels_tensor = torch.from_numpy(labels).float()

# Add batch and channel dimensions for interpolate: (N, C, D, H, W)
labels_tensor = labels_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)

# Resize labels to the same dimensions as the voxel data
labels_tensor = torch.nn.functional.interpolate(labels_tensor, size=dims, mode='nearest')

# Convert to int32
labels_tensor = labels_tensor.int()

# Remove batch and channel dimensions and convert back to numpy array
labels_tensor = labels_tensor.squeeze(0).squeeze(0).numpy().astype(np.int32)

labels_numpy = np.asarray(labels_tensor)

# Save resampled labels
np.savez_compressed(os.path.join(parent_dir, 'resampled_labels.npz'), data=labels_numpy)

print('Resampled labels saved to: ', os.path.join(parent_dir, 'resampled_labels.npz'))