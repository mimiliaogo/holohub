# Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

#
# A tool to convert legacy RenderServer settings to RenderServer NG settings
#

import os
import argparse
import json
import math
import numpy as np

def convertFloat3(array, scale = 1.0):
    return { 'x': array[0] * scale,
        'y': array[1] * scale,
        'z': array[2] * scale }

def convertColor3(array):
    return { 'x': array[0] / 255.0,
        'y': array[1] / 255.0,
        'z': array[2] / 255.0 }

def convertEnable(enable):
    return enable
    #return 'SWITCH_ENABLE' if (enable) else 'SWITCH_DISABLE'

def convertTransferFunctionComponents(dict_tf_components, active_regions):
    components = []
    for dict_component in dict_tf_components:
        if not dict_component['enabled']:
            continue

        component = {}
        component['range'] = { 'min': dict_component['left'], 'max': dict_component['right'] }

        component['activeRegions'] = []
        component['activeRegions'].extend(active_regions)
        # compatibility with old format w/o presets
        if 'active_labels' in dict_component:
            component['activeRegions'].extend(dict_component['active_labels'])

        conv_opacity_profile = {
            'Square' : 'SQUARE',
            'Triangle' : 'TRIANGLE',
            'Sine' : 'SINE',
            'Trapeziod' : 'TRAPEZIOD'
        }
        component['opacityProfile'] = conv_opacity_profile.get(dict_component['opacityProfile'], 'OPACITY_PROFILE_UNKNOWN')
        component['opacityTransition'] = dict_component['transition']
        component['opacity'] = dict_component['opacity']
        component['roughness'] = dict_component['roughness']
        component['emissiveStrength'] = dict_component['emissive']
        component['diffuseStart'] = convertColor3(dict_component['diffuseStart'])
        component['diffuseEnd'] = convertColor3(dict_component['diffuseEnd'])
        component['specularStart'] = convertColor3(dict_component['specularStart'])
        component['specularEnd'] = convertColor3(dict_component['specularEnd'])
        component['emissiveStart'] = convertColor3(dict_component['emissiveStart'])
        component['emissiveEnd'] = convertColor3(dict_component['emissiveEnd'])

        components.append(component)

    return components

def convert(args, volume_size):
    in_dict = json.load(args.json)

    out_dict = {}

    dict_background_light = in_dict['light_settings']['backgroundlight']
    out_dict['BackgroundLight'] = {}
    out_background_light = out_dict['BackgroundLight']
    out_background_light['topColor'] = convertColor3(dict_background_light['topColor'])
    out_background_light['horizonColor'] = convertColor3(dict_background_light['middleColor'])
    out_background_light['bottomColor'] = convertColor3(dict_background_light['bottomColor'])
    out_background_light['intensity'] = dict_background_light['intensity']
    out_background_light['enable'] = convertEnable(dict_background_light['enabled'])
    out_background_light['castLight'] = convertEnable(dict_background_light['castLight'])
    out_background_light['show'] = convertEnable(dict_background_light['show'])

    dict_camera = in_dict['camera_settings']
    out_dict['Camera'] = {}
    out_camera = out_dict['Camera']
    out_camera['eye'] = convertFloat3(dict_camera['eye'], volume_size)
    out_camera['lookAt'] = convertFloat3(dict_camera['lookat'], volume_size)
    out_camera['up'] = convertFloat3(dict_camera['up'])
    out_camera['fieldOfView'] = dict_camera['fov']
    out_camera['pixelAspectRatio'] = 1

    out_dict['CameraAperture'] = {}
    out_camera_aperture = out_dict['CameraAperture']
    out_camera_aperture['autoFocus'] = convertEnable(dict_camera['autoFocus'])
    out_camera_aperture['focusDistance'] = dict_camera['focus']
    if dict_camera['aperture'] > 0:
        out_camera_aperture['aperture'] = dict_camera['aperture']
        out_camera_aperture['enable'] = convertEnable(True)

    dict_lights = in_dict['light_settings']['lights']
    out_dict['Light'] = []
    index = 0
    for dict_light in dict_lights:
        out_dict['Light'].append({})
        out_light = out_dict['Light'][index]
        out_light['index'] = index

        if 'zenith' in dict_light:
            distance = dict_light['distance'] * volume_size
            # legacy RenderServer used spherical coordinates, convert to cartesian coordinates
            position = np.array([
                distance * math.cos(dict_light['zenith'] * math.pi / 180.0) * math.sin(dict_light['azimuth'] * math.pi / 180.0),
                distance * math.cos(dict_light['zenith'] * math.pi / 180.0) * math.cos(dict_light['azimuth'] * math.pi / 180.0),
                distance * math.sin(dict_light['zenith'] * math.pi / 180.0)])
            out_light['position'] = { 'x': position[0],
                'y': position[1],
                'z': position[2] }
            direction = np.array([0.0, 0.0, 0.0]) - position
            direction /= np.linalg.norm(direction)
            out_light['direction'] = { 'x': direction[0],
                'y': direction[1],
                'z': direction[2] }
        else:
            out_light['position'] = convertFloat3(dict_light['position'], volume_size)
            out_light['direction'] = convertFloat3(dict_light['direction'])

        out_light['size'] = dict_light['size'] * volume_size
        out_light['intensity'] = dict_light['intensity'] * volume_size * volume_size
        out_light['color'] = convertColor3(dict_light['color'])
        out_light['enable'] = convertEnable(dict_light['enabled'])
        out_light['show'] = convertEnable(dict_light['show'])

        index += 1

    dict_post_process = in_dict['post_process_settings']
    out_dict['PostProcessDenoise'] = {}
    out_post_process_denoise = out_dict['PostProcessDenoise']
    out_post_process_denoise['method'] = 'KNN' if (dict_post_process['denoiseEnabled']) else 'OFF'
    out_post_process_denoise['radius'] = dict_post_process['denoiseWindowRadius']
    out_post_process_denoise['spatialWeight'] = dict_post_process['denoiseSpatialWeight']
    out_post_process_denoise['depthWeight'] = dict_post_process['denoiseDepthWeight']
    out_post_process_denoise['noiseThreshold'] = dict_post_process['denoiseNoiseThreshold']
    out_post_process_denoise['enableIterationLimit'] = convertEnable(dict_post_process['denoiseLimitEnabled'])
    out_post_process_denoise['iterationLimit'] = dict_post_process['denoiseIterationLimit']

    out_dict['PostProcessTonemap'] = {}
    out_post_process_tonemap = out_dict['PostProcessTonemap']
    out_post_process_tonemap['enable'] = convertEnable(dict_post_process['tonemapEnabled'])
    out_post_process_tonemap['exposure'] = dict_post_process['tonemapExposure']

    dict_render_settings = in_dict['render_settings']
    out_dict['RenderSettings'] = {}
    out_render_settings = out_dict['RenderSettings']
    conv_interpolation_mode = {
        'Linear' : 'LINEAR',
        'B Spline' : 'BSPLINE',
        'CatmullRom Spline' : 'CATMULLROM',
    }
    out_render_settings['interpolationMode'] = conv_interpolation_mode.get(dict_render_settings['interpolationMode'], 'LINEAR')
    out_render_settings['stepSize'] = dict_render_settings['stepSize']
    out_render_settings['shadowStepSize'] = dict_render_settings['shadowStepSize']
    out_render_settings['maxIterations'] = dict_render_settings['maxIterations']
    out_render_settings['timeSlot'] = dict_render_settings['timeSlot']
    out_render_settings['enableWarp'] = convertEnable(dict_render_settings['useWarp'])
    out_render_settings['warpResolutionScale'] = dict_render_settings['warpResolutionScale']
    out_render_settings['warpFullResolutionSize'] = dict_render_settings['warpFullResolutionDiameter']
    out_render_settings['enableFoveation'] = convertEnable(dict_render_settings['useFoveation'])
    out_render_settings['enableReproject'] = convertEnable(dict_render_settings['useWarp'])

    dict_tf_settings = in_dict['transfer_function_settings']['settings']
    out_dict['TransferFunction'] = {}
    out_transfer_function = out_dict['TransferFunction']
    conv_shading_profile = {
        'Hybrid' : 'HYBRID',
        'BRDF Only' : 'BRDF_ONLY',
        'Phase Function Only' : 'PHASE_ONLY'
    }
    out_transfer_function['shadingProfile'] = conv_shading_profile.get(dict_tf_settings['shadingProfile'], 'SHADING_PROFILE_UNKNOWN')
    conv_blending_profile = {
        'Maximum Opacity' : 'MAXIMUM_OPACITY',
        'Blended Opacity' : 'BLENDED_OPACITY'
    }
    out_transfer_function['blendingProfile'] = conv_blending_profile.get(dict_tf_settings['blendingProfile'], 'BLENDING_PROFILE_UNKNOWN')
    out_transfer_function['globalOpacity'] = dict_tf_settings['globalOpacity']
    out_transfer_function['densityScale'] = dict_tf_settings['densityFactor']
    out_transfer_function['gradientScale'] = dict_tf_settings['gradientFactor']

    out_transfer_function['components'] = []

    # check for the new 'presets'
    if 'presets' in in_dict['transfer_function_settings']:
        dict_tf_presets = in_dict['transfer_function_settings']['presets']
        for dict_tf_preset in dict_tf_presets:
            if not dict_tf_preset['enabled']:
                continue

            out_transfer_function['components'].extend(convertTransferFunctionComponents(dict_tf_preset['components'],
                dict_tf_preset['activeRegions']))
    else:
        out_transfer_function['components'].extend(convertTransferFunctionComponents(in_dict['transfer_function_settings']['components'], []))

    dict_volume = in_dict['volume_settings']

    out_dict['VolumeCrop'] = {}
    out_volume_crop = out_dict['VolumeCrop']
    out_volume_crop['min'] = { 'x': dict_volume['cropXStart'], 'y': dict_volume['cropYStart'], 'z': dict_volume['cropZStart'] }
    out_volume_crop['max'] = { 'x': dict_volume['cropXEnd'], 'y': dict_volume['cropYEnd'], 'z': dict_volume['cropZEnd'] }

    # copy OpenXR POC dataset section
    if 'dataset' in in_dict:
        out_dict['dataset'] = in_dict['dataset']

    # if the output is a directory write multiple files, else one single file
    if os.path.isdir(args.output):
        for key, values in out_dict.items():
            if isinstance(values, list):
                index = 0
                for value in values:
                    with open(os.path.join(args.output, key + str(index) + '.json'), 'w') as out_file:
                        json.dump(value, out_file, sort_keys=True, indent=1)
                    index += 1
            else:
                with open(os.path.join(args.output, key + '.json'), 'w') as out_file:
                    json.dump(values, out_file, sort_keys=True, indent=1)
    else:
        with open(args.output, 'w') as out_file:
            json.dump(out_dict, out_file, sort_keys=True, indent=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert legacy RenderServer settings to RenderServer NG settings.')

    parser.add_argument('--json', type=argparse.FileType('r'), help='legacy RenderServer settings JSON file')
    parser.add_argument('--mhd', type=argparse.FileType('r'), help='MHD file, used to scale camera and lights')
    parser.add_argument('-o', '--output', dest='output', default='.', help='Output. If this is a directory multiple files will be stored, if this is a file name all settings will be written to that single file.')

    args = parser.parse_args()

    # Legacy RenderServer internally scaled camera and light by the length of the diagonal of the volume bounding box
    # Since RenderServer NG no longer is doing that, scale camera and light while converting the settings.
    if args.mhd:
        dim_size = [1, 1, 1]
        element_spacing = [0.001, 0.001, 0.001]
        print('Reading volume config')
        for line in args.mhd:
            items = line.split()
            if items[0] == 'DimSize':
                dim_size[0] = int(items[2])
                dim_size[1] = int(items[3])
                dim_size[2] = int(items[4])
            if items[0] == 'ElementSpacing':
                element_spacing[0] = float(items[2]) * 0.001
                element_spacing[1] = float(items[3]) * 0.001
                element_spacing[2] = float(items[4]) * 0.001
        print(f'dim size {dim_size}, element spacing {element_spacing}')

        volume_size = math.sqrt(
            math.pow(dim_size[0] * element_spacing[0], 2) +
            math.pow(dim_size[1] * element_spacing[1], 2) +
            math.pow(dim_size[2] * element_spacing[2], 2))
        print (f'Scaling camera lookat/eye and light position/size/intensity by {volume_size}')
    else:
        volume_size = 1.0

    convert(args, volume_size)
