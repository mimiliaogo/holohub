# BCI Visualization with ClaraViz

This application demonstrates real-time visualization of Brain-Computer Interface (BCI) data using the Holoscan SDK and ClaraViz volume rendering - **pure Python implementation**.

## Overview

The application creates a pipeline that:
1. Generates synthetic voxel-wise hemodynamic data (HbO/HbR) using a test loader
2. Converts streaming voxel data into a dense 3D volume using a bridge operator
3. Renders the volume in real-time using ClaraViz
4. Provides interactive visualization with Holoviz

This architecture is designed to support real-time BCI data visualization where sparse voxel measurements are continuously streamed and rendered as a volumetric brain image.

## Architecture

### Components

All components are implemented in **pure Python**:

1. **SimpleVoxelLoaderOp** (`simple_voxel_loader.py`): Generates synthetic voxel data
   - Outputs: `affine_4x4`, `voxel_xyz`, `hb_frame`
   - Simulates a BCI data source with time-varying hemodynamic signals

2. **VoxelStreamToVolumeOp** (`voxel_stream_to_volume.py`): Bridge operator that converts sparse voxel data to dense volume
   - Inputs: `affine_4x4`, `voxel_xyz`, `hb_frame`
   - Outputs: `volume`, `spacing`, `permute_axis`, `flip_axes`
   - Performs spatial interpolation and grid mapping using NumPy and CuPy

3. **VolumeRendererOp**: ClaraViz-based volume renderer
   - Renders the 3D volume with transfer functions
   - Supports interactive camera control

4. **HolovizOp**: Interactive visualization
   - Displays the rendered volume
   - Provides camera pose feedback

## Installation

Requires:
- Holoscan SDK (>= 0.6.0)
- Python 3.8+
- NumPy
- CuPy (for GPU acceleration)

```bash
pip install numpy cupy-cuda11x  # or cupy-cuda12x depending on your CUDA version
```

## Running

Simple usage (no command-line arguments needed):
```bash
cd /home/mimil/Projects/holohub/applications/bci_visualization
python bci_visualization.py
```

Or from holohub root:
```bash
python applications/bci_visualization/bci_visualization.py
```

## Configuration

All parameters are hardcoded for simplicity in `bci_visualization.py`:
- **Grid dimensions**: 160 x 160 x 96
- **Voxel spacing**: 1.0 x 1.0 x 1.0 mm
- **Number of test voxels**: 1000
- **Channels**: 2 (HbO/HbR)
- **Selected channel**: 0 (HbO)
- **Renderer config**: `../../data/volume_rendering/config.json`

To customize these values, edit the hardcoded constants in `bci_visualization.py`'s `compose()` method.

## Data Flow

```
SimpleVoxelLoader → VoxelStreamToVolume → VolumeRenderer → Holoviz
                                             ↑               ↓
                                             └─── camera ────┘
```

## Files

- `bci_visualization.py` - Main application
- `simple_voxel_loader.py` - Test data generator operator
- `voxel_stream_to_volume.py` - Bridge operator (voxel → volume)
- `prepare_voxel_from_nifti.py` - Utility to prepare voxel data from NIfTI files

## Future Enhancements

This test application can be extended to:
- Replace `SimpleVoxelLoaderOp` with a real BCI data source operator
- Add support for multiple channels visualization
- Implement advanced interpolation methods (trilinear, gaussian)
- Add real-time signal processing operators
- Load real voxel positions from NIfTI files using `prepare_voxel_from_nifti.py`

## License

SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
SPDX-License-Identifier: Apache-2.0
