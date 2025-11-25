import nibabel as nib
import numpy as np
import argparse
import sys


def extract_frame_from_4d_nifti(input_path, output_path, frame_index):
    """
    Extracts a single frame from a 4D NIfTI file and saves it as a 3D NIfTI file.
    
    Args:
        input_path (str): Path to the input 4D .nii.gz file
        output_path (str): Path to save the extracted 3D .nii.gz file
        frame_index (int): Index of the frame to extract (0-based)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load the NIfTI file
        print(f"Loading NIfTI file: {input_path}")
        img = nib.load(input_path)
        data = img.get_fdata()
        
        print(f"Data shape: {data.shape}")
        print(f"Data dimensions: {data.ndim}D")
        
        # Check if this is a 4D volume
        if data.ndim != 4:
            print(f"Error: Input file is {data.ndim}D, expected 4D volume")
            return False
        
        num_frames = data.shape[-1]
        print(f"Number of frames: {num_frames}")
        
        # Validate frame index
        if frame_index < 0 or frame_index >= num_frames:
            print(f"Error: Frame index {frame_index} is out of range [0, {num_frames-1}]")
            return False
        
        # Extract the specified frame
        print(f"Extracting frame {frame_index}...")
        frame_data = data[..., frame_index]
        
        print(f"Extracted frame shape: {frame_data.shape}")
        print(f"Frame data range: [{np.min(frame_data):.4f}, {np.max(frame_data):.4f}]")
        
        # Convert to float32 for compatibility with Holoscan volume renderer
        # (NIfTI loader only supports up to FLOAT32, not FLOAT64)
        if frame_data.dtype != np.float32:
            print(f"Converting from {frame_data.dtype} to float32...")
            frame_data = frame_data.astype(np.float32)
        
        # Create a new 3D NIfTI image
        # Important: Preserve the affine transformation from the original image
        new_img = nib.Nifti1Image(frame_data, img.affine)
        
        # Copy relevant header information
        new_header = new_img.header
        orig_header = img.header
        
        # Copy voxel dimensions (spacing) - first 3 dimensions only
        new_header['pixdim'][1:4] = orig_header['pixdim'][1:4]
        
        # Copy data type
        new_header.set_data_dtype(frame_data.dtype)
        
        # Save the 3D NIfTI image
        nib.save(new_img, output_path)
        
        print(f"\nSuccessfully extracted frame {frame_index} from '{input_path}'")
        print(f"Saved to: '{output_path}'")
        
        return True
        
    except FileNotFoundError:
        print(f"Error: The file '{input_path}' was not found.")
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        return False


def list_frame_info(input_path):
    """
    Lists information about all frames in a 4D NIfTI file.
    
    Args:
        input_path (str): Path to the input 4D .nii.gz file
    """
    try:
        print(f"Loading NIfTI file: {input_path}")
        img = nib.load(input_path)
        data = img.get_fdata()
        
        print(f"\nData shape: {data.shape}")
        print(f"Data dimensions: {data.ndim}D")
        
        if data.ndim != 4:
            print(f"Warning: Input file is {data.ndim}D, expected 4D volume")
            return
        
        num_frames = data.shape[-1]
        print(f"Number of frames: {num_frames}")
        print(f"Spatial dimensions: {data.shape[0]}x{data.shape[1]}x{data.shape[2]}")
        print(f"\nFrame statistics:")
        print("-" * 60)
        
        for frame_idx in range(num_frames):
            frame = data[..., frame_idx]
            frame_min = np.min(frame)
            frame_max = np.max(frame)
            frame_mean = np.mean(frame)
            frame_std = np.std(frame)
            
            print(f"Frame {frame_idx:3d}: min={frame_min:10.4f}, max={frame_max:10.4f}, "
                  f"mean={frame_mean:10.4f}, std={frame_std:10.4f}")
        
    except Exception as e:
        print(f"An error occurred: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract a single frame from a 4D NIfTI file and save as 3D NIfTI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract frame 5 from a 4D volume
  python extract_frame.py input_4d.nii.gz output_3d.nii.gz --frame 5
  
  # List information about all frames
  python extract_frame.py input_4d.nii.gz --info
        """
    )
    
    parser.add_argument('input', help='Input 4D NIfTI file path')
    parser.add_argument('output', nargs='?', help='Output 3D NIfTI file path')
    parser.add_argument('--frame', '-f', type=int, default=0,
                        help='Frame index to extract (0-based, default: 0)')
    parser.add_argument('--info', '-i', action='store_true',
                        help='List information about all frames and exit')
    
    args = parser.parse_args()
    
    # If --info flag is set, just list frame information
    if args.info:
        list_frame_info(args.input)
        return
    
    # Otherwise, extract the frame
    if not args.output:
        print("Error: output path is required when not using --info")
        parser.print_help()
        sys.exit(1)
    
    success = extract_frame_from_4d_nifti(args.input, args.output, args.frame)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    # If running directly (not via command line), use these defaults
    if len(sys.argv) == 1:
        # --- USAGE EXAMPLE ---
        frame_index = 55
        input_file = 'data/volume_rendering/study-trustreliabilitystudy_sub-trust1023s175_desc-563862f_HbO_normalized.nii.gz'
        output_file = f'data/volume_rendering/study-trustreliabilitystudy_sub-trust1023s175_desc-563862f_HbO_normalized_frame_{frame_index}.nii.gz'
        
        print("Running with default parameters:")
        print(f"  Input: {input_file}")
        print(f"  Output: {output_file}")
        print(f"  Frame: {frame_index}\n")
        
        extract_frame_from_4d_nifti(input_file, output_file, frame_index)
    else:
        main()

