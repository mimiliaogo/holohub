import nibabel as nib
import numpy as np

def rescale_nifti_to_ct_range(input_path, output_path, frame_by_frame=True):
    """
    Rescales a .nii.gz file's data from its original range to a new
    target range, such as the typical CT range of [-1024, 3071].
    
    For 4D volumes (with time dimension), can normalize either:
    - frame-by-frame: Each time frame is normalized independently based on its own min/max
    - globally: All frames normalized using global min/max across entire 4D volume

    Args:
        input_path (str): The path to the input .nii.gz file.
        output_path (str): The path to save the rescaled .nii.gz file.
        frame_by_frame (bool): If True, normalize each frame independently (4D only).
                               If False, use global min/max for all frames.
    """
    # Define the new target range
    new_min = -1024.0
    new_max = 3071.0

    try:
        # Load the NIfTI file
        img = nib.load(input_path)
        data = img.get_fdata()
        
        print(f"Data shape: {data.shape}")
        print(f"Data dimensions: {data.ndim}D")

        # Check if this is a 4D volume (time series)
        is_4d = data.ndim == 4
        
        if is_4d and frame_by_frame:
            print("Performing frame-by-frame normalization...")
            num_frames = data.shape[-1]
            rescaled_data = np.zeros_like(data)
            
            for frame_idx in range(num_frames):
                frame = data[..., frame_idx]
                
                # Get the frame's min and max values
                frame_min = np.min(frame)
                frame_max = np.max(frame)
                
                print(f"Frame {frame_idx}: original range [{frame_min:.4f}, {frame_max:.4f}]", end="")
                
                # Check for constant data to avoid division by zero
                if frame_max == frame_min:
                    print(" -> constant data, filling with new_min")
                    rescaled_data[..., frame_idx] = new_min
                else:
                    # Apply the Min-Max scaling formula
                    rescaled_frame = new_min + ((frame - frame_min) * (new_max - new_min)) / (frame_max - frame_min)
                    rescaled_data[..., frame_idx] = rescaled_frame
                    
                    rescaled_min = np.min(rescaled_frame)
                    rescaled_max = np.max(rescaled_frame)
                    print(f" -> rescaled range [{rescaled_min:.2f}, {rescaled_max:.2f}]")
        else:
            # Global normalization (3D or 4D with frame_by_frame=False)
            if is_4d:
                print("Performing global normalization across all frames...")
            else:
                print("Performing global normalization...")
                
            # Get the original data's min and max values
            original_min = np.min(data)
            original_max = np.max(data)
            
            print(f"Original data range: [{original_min:.4f}, {original_max:.4f}]")

            # Check for constant data to avoid division by zero
            if original_max == original_min:
                print("Image data is constant; rescaling is not applicable.")
                rescaled_data = np.full(data.shape, new_min)
            else:
                # Apply the Min-Max scaling formula for an arbitrary range [a, b]
                # X_scaled = a + ((X - X_min) * (b - a)) / (X_max - X_min)
                rescaled_data = new_min + ((data - original_min) * (new_max - new_min)) / (original_max - original_min)
                print(f"New data range: [{np.min(rescaled_data):.2f}, {np.max(rescaled_data):.2f}]")

                # statistics
                # convert to 0-1 range
                rescaled_data_01 = (rescaled_data - rescaled_data.min()) / (rescaled_data.max() - rescaled_data.min())
                # print out unique values and their counts
                unique_values, counts = np.unique(rescaled_data_01, return_counts=True)
                print(unique_values, counts)
                # print out counts > 10
                print(unique_values[counts > 1000000])


        # Create a new NIfTI image with the rescaled data
        # IMPORTANT: Reuse the original affine and header to preserve spatial info
        new_img = nib.Nifti1Image(rescaled_data, img.affine, img.header)

        # Save the new NIfTI image
        nib.save(new_img, output_path)
        
        print(f"\nSuccessfully rescaled '{input_path}' and saved to '{output_path}'")

    except FileNotFoundError:
        print(f"Error: The file '{input_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# --- USAGE ---
# Replace with your file paths
input_file = 'data/volume_rendering/filtered_activation_volume.nii.gz'
output_file = 'data/volume_rendering/filtered_activation_volume_normalized.nii.gz'

# Option 1: Frame-by-frame normalization (each frame normalized independently)
# This is useful when each time frame has different intensity ranges
# rescale_nifti_to_ct_range(input_file, output_file, frame_by_frame=True)

# Option 2: Global normalization (all frames use same min/max)
# Uncomment below to use global normalization instead
rescale_nifti_to_ct_range(input_file, output_file, frame_by_frame=False)