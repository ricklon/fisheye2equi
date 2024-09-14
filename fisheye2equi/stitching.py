# fisheye2equi/stitching.py

import numpy as np
import cv2
import logging
import os
import re

logger = logging.getLogger(__name__)

def load_pto_profile(profile_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    profile_path = os.path.join(current_dir, 'profiles', profile_name)
    if not os.path.exists(profile_path):
        raise FileNotFoundError(f"PTO profile not found: {profile_path}")
    
    profile_data = {}
    with open(profile_path, 'r') as f:
        content = f.read()
        
        # Extract panorama size
        size_match = re.search(r'p f2 w(\d+) h(\d+) v(\d+)', content)
        if size_match:
            profile_data['panorama_width'] = int(size_match.group(1))
            profile_data['panorama_height'] = int(size_match.group(2))
            profile_data['panorama_fov'] = int(size_match.group(3))
        
        # Extract image size and field of view for both lenses
        image_matches = re.findall(r'i w(\d+) h(\d+) f2 v([\d.]+)', content)
        if len(image_matches) == 2:
            profile_data['image_width'] = int(image_matches[0][0])
            profile_data['image_height'] = int(image_matches[0][1])
            profile_data['fov_left'] = float(image_matches[0][2])
            profile_data['fov_right'] = float(image_matches[1][2])
        
        # Extract distortion parameters (a, b, c, d, e)
        distortion_matches = re.findall(r'a([\d.-]+) b([\d.-]+) c([\d.-]+) d([\d.-]+) e([\d.-]+)', content)
        if distortion_matches:
            profile_data['distortion_left'] = [float(x) for x in distortion_matches[0]]
            profile_data['distortion_right'] = [float(x) for x in distortion_matches[1]]
        else:
            # If no distortion parameters found, use zeros
            profile_data['distortion_left'] = [0, 0, 0, 0]
            profile_data['distortion_right'] = [0, 0, 0, 0]
        
        # Extract rotation parameters
        rotation_matches = re.findall(r'r(-?[\d.]+) p(-?[\d.]+) y(-?[\d.]+)', content)
        if rotation_matches:
            # Swap the order to [yaw, pitch, roll]
            rotation_left = [float(rotation_matches[0][2]), float(rotation_matches[0][1]), float(rotation_matches[0][0])]
            rotation_right = [float(rotation_matches[1][2]), float(rotation_matches[1][1]), float(rotation_matches[1][0])]
            profile_data['rotation_left'] = rotation_left
            profile_data['rotation_right'] = rotation_right
        else:
            profile_data['rotation_left'] = [0, 0, 0]
            profile_data['rotation_right'] = [0, 0, 0]
            
        logger.debug(f"Loaded rotation values: Left {profile_data['rotation_left']}, Right {profile_data['rotation_right']}")
        
        return profile_data

def fisheye_to_equirectangular(fisheye_img, width, height, fov, distortion_params, rotation):
    logger.info(f"Converting fisheye to equirectangular with rotation: width={width}, height={height}, fov={fov}, rotation={rotation}")
    
    # Ensure input image is float32
    fisheye_img = np.float32(fisheye_img)
    
    # Fisheye image dimensions
    h_fisheye, w_fisheye = fisheye_img.shape[:2]
    cx, cy = w_fisheye / 2.0, h_fisheye / 2.0  # Center of the fisheye image

    # Generate grid for equirectangular image
    theta = np.linspace(-np.pi, np.pi, width)
    phi = np.linspace(np.pi / 2, -np.pi / 2, height)  # From top to bottom
    theta, phi = np.meshgrid(theta, phi)

    # Spherical to Cartesian coordinates (unit sphere)
    x = np.cos(phi) * np.cos(theta)
    y = np.cos(phi) * np.sin(theta)
    z = np.sin(phi)

    # Flatten the arrays
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()

    # Stack into 3 x N array
    xyz = np.vstack((x, y, z))

    # Convert rotation angles to radians
    ryaw, rpitch, rroll = np.radians(rotation)

    # Create rotation matrices
    Rz = np.array([
        [np.cos(ryaw), -np.sin(ryaw), 0],
        [np.sin(ryaw),  np.cos(ryaw), 0],
        [0,             0,            1]
    ], dtype=np.float32)

    Ry = np.array([
        [np.cos(rpitch), 0, np.sin(rpitch)],
        [0,              1, 0],
        [-np.sin(rpitch), 0, np.cos(rpitch)]
    ], dtype=np.float32)

    Rx = np.array([
        [1, 0,             0],
        [0, np.cos(rroll), -np.sin(rroll)],
        [0, np.sin(rroll),  np.cos(rroll)]
    ], dtype=np.float32)

    # Combined rotation matrix
    R = Rz @ Ry @ Rx

    # Apply rotation
    xyz_rotated = R @ xyz

    # Rotated coordinates
    x_rot, y_rot, z_rot = xyz_rotated

    # Optionally invert axes if needed
    # y_rot = -y_rot  # Commented out
    # z_rot = -z_rot  # Uncomment if needed

    # Convert back to spherical coordinates
    theta_fisheye = np.arctan2(y_rot, x_rot)
    phi_fisheye = np.arccos(z_rot / np.sqrt(x_rot**2 + y_rot**2 + z_rot**2))

    # Focal length for fisheye projection
    f = w_fisheye / (2.0 * np.tan(np.radians(fov / 2.0)))

    # Radial distance from center (equisolid angle projection)
    r = 2 * f * np.sin(phi_fisheye / 2)

    # Fisheye image coordinates
    x_fisheye = cx + r * np.cos(theta_fisheye)
    y_fisheye = cy + r * np.sin(theta_fisheye)

    # Reshape back to image shape
    map_x = x_fisheye.reshape((height, width)).astype(np.float32)
    map_y = y_fisheye.reshape((height, width)).astype(np.float32)

    # Mask invalid coordinates
    mask = (map_x >= 0) & (map_x < w_fisheye) & (map_y >= 0) & (map_y < h_fisheye)

    # Use remap to generate the equirectangular image
    equirectangular = cv2.remap(fisheye_img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Set invalid regions to black
    equirectangular[~mask] = 0

    return equirectangular






def stitch_simple(equi_left, equi_right, width, height):
    logger.info(f"Stitching equirectangular pair using simple method: width={width}, height={height}")

    # Ensure input images are float32 and in [0, 1] range
    equi_left = np.clip(np.float32(equi_left), 0, 1)
    equi_right = np.clip(np.float32(equi_right), 0, 1)

    # Initialize the stitched image
    stitched_equirectangular = np.zeros((height, width, 3), dtype=np.float32)

    # Compute the overlap region
    overlap_width = width // 32  # Adjust as necessary
    half_width = width // 2

    # Left non-overlapping region
    stitched_equirectangular[:, :half_width - overlap_width] = equi_left[:, :half_width - overlap_width]

    # Right non-overlapping region
    stitched_equirectangular[:, half_width + overlap_width:] = equi_right[:, overlap_width:]

    # Overlapping regions
    left_overlap = equi_left[:, half_width - overlap_width: half_width]
    right_overlap = equi_right[:, :overlap_width]

    # Create alpha blending mask
    alpha = np.linspace(1, 0, overlap_width)[np.newaxis, :, np.newaxis]  # Shape (1, overlap_width, 1)
    alpha = np.repeat(alpha, height, axis=0)  # Shape (height, overlap_width, 1)

    # Blend the overlapping regions
    blended_overlap = left_overlap * alpha + right_overlap * (1 - alpha)

    # Place blended overlap into the stitched image
    stitched_equirectangular[:, half_width - overlap_width: half_width] = blended_overlap

    # Place the rest of the right image
    stitched_equirectangular[:, half_width:] = equi_right[:, :half_width]

    # Ensure output is in [0, 1] range
    stitched_equirectangular = np.clip(stitched_equirectangular, 0, 1)

    return stitched_equirectangular



def stitch_gear360_image(gear360_image, pto_profile, image_info, stitching_method='simple', debug=False):
    debug_images = {}
    try:
        # Ensure input image is float32 and in [0, 1] range
        gear360_image = np.clip(np.float32(gear360_image) / 255.0, 0, 1)
        
        if debug:
            debug_images['Original Image'] = (gear360_image * 255).astype(np.uint8)
        
        # Load the .pto profile
        logger.debug(f"Loading PTO profile: {pto_profile}")
        profile_data = load_pto_profile(pto_profile)
        logger.debug(f"Profile data loaded: {profile_data}")
        
        # Split the image into left and right fisheye images
        h, w = gear360_image.shape[:2]
        fisheye_left = gear360_image[:, :w // 2]
        fisheye_right = gear360_image[:, w // 2:]
        logger.debug(f"Image split into left and right. Dimensions: {fisheye_left.shape}, {fisheye_right.shape}")

        if debug:
            debug_images['Left Fisheye'] = (fisheye_left * 255).astype(np.uint8)
            debug_images['Right Fisheye'] = (fisheye_right * 255).astype(np.uint8)

        # Use profile data for output dimensions and FOV
        output_width = profile_data['panorama_width']
        output_height = profile_data['panorama_height']
        fov_left = 180.0  # Use standard FOV
        fov_right = 180.0

        logger.info(f"Output dimensions: {output_width}x{output_height}, FOV: Left {fov_left}, Right {fov_right}")

        logger.info("Converting fisheye to equirectangular with rotation")
        rotation_left = profile_data['rotation_left']
        rotation_right = profile_data['rotation_right']

        # Adjust rotations
        # Experiment with different adjustments
        rotation_left[0] += 0    # No adjustment to left yaw
        rotation_right[0] += 180  # Add 180 degrees to right yaw

        # Ensure angles are within [-180, 180]
        rotation_left[0] = (rotation_left[0] + 180) % 360 - 180
        rotation_right[0] = (rotation_right[0] + 180) % 360 - 180

        # Proceed with generating equirectangular images
        equi_left = fisheye_to_equirectangular(
            fisheye_left,
            output_width // 2,
            output_height,
            fov_left,
            profile_data['distortion_left'],
            rotation_left
        )

        equi_right = fisheye_to_equirectangular(
            fisheye_right,
            output_width // 2,
            output_height,
            fov_right,
            profile_data['distortion_right'],
            rotation_right
        )
    
        logger.debug(f"Equirectangular conversion complete. Dimensions: {equi_left.shape}, {equi_right.shape}")

        if debug:
            debug_images['Equirectangular Left'] = (equi_left * 255).astype(np.uint8)
            debug_images['Equirectangular Right'] = (equi_right * 255).astype(np.uint8)

        logger.info(f"Stitching equirectangular images using {stitching_method} method")
        stitch_func = stitch_simple  # Currently, only simple method is implemented
        stitched_equirectangular = stitch_func(equi_left, equi_right, output_width, output_height)

        if stitched_equirectangular is None:
            logger.warning("Stitching failed, returning the left equirectangular image")
            return (equi_left * 255).astype(np.uint8), debug_images

        # Convert back to uint8 for display/saving
        result = (np.clip(stitched_equirectangular, 0, 1) * 255).astype(np.uint8)
        
        if debug:
            debug_images['Final Stitched Image'] = result

        return result, debug_images

    except Exception as e:
        logger.error(f"An error occurred during stitching: {str(e)}", exc_info=True)
        return (gear360_image * 255).astype(np.uint8), debug_images
