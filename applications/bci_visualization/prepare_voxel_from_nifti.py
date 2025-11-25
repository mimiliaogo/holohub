import numpy as np
import nibabel as nib
from nilearn import image
import matplotlib.pyplot as plt
import os

parent_dir = "/workspace/holohub/data/bci_visualization"
# --- 1. Load Data ---
print("Loading data...")

# Load 3D anatomical volume (same as before)
anatomical_volume = nib.load(os.path.join(parent_dir, 'volume.nii.gz'))
print(f"Anatomical shape (3D): {anatomical_volume.shape}")

activation_volume_4d = nib.load(os.path.join(parent_dir, 'study-trustreliabilitystudy_sub-trust1023s175_desc-563862f_HbO.nii.gz'))
print(f"Loaded 4D activation shape: {activation_volume_4d.shape}")
# print range for the first time point (frame 0)
print(f'Loaded 4D volume range (frame 0): {activation_volume_4d.slicer[..., 0].get_fdata().min()} to {activation_volume_4d.slicer[..., 0].get_fdata().max()}')

print('---')

# Extract only 10 time points
activation_volume_4d = activation_volume_4d.slicer[..., 250:255]

# --- 2. Resample 4D Volume ---
print("Resampling 4D volume to anatomical space...")
# nilearn's resample_to_img handles 4D inputs correctly.
# It resamples each 3D time-point to the target_img's spatial grid.
resampled_activation_volume_4d = image.resample_to_img(
    source_img=activation_volume_4d,
    target_img=anatomical_volume,
    interpolation='continuous'
)
# save resampled 4D volume
nib.save(resampled_activation_volume_4d, os.path.join(parent_dir, 'resampled_activation_volume_4D.nii.gz'))
# The new shape will be (anat_x, anat_y, anat_z, time)
print(f"Resampled 4D volume shape: {resampled_activation_volume_4d.shape}")