# stitching.py

import os
import numpy as np
import cv2
import re
import logging

logger = logging.getLogger(__name__)

def load_pto_profile(profile_name):
    profile_data = {}
    try:
        # Build the path to the PTO file
        script_dir = os.path.dirname(__file__)  # Directory of the current script
        profile_path = os.path.join(script_dir, 'profiles', profile_name)

        logger.debug(f"Loading PTO profile from: {profile_path}")

        with open(profile_path, 'r') as f:
            content = f.read()
        
        # Corrected regex to handle negative numbers
        rotation_matches = re.findall(r'r(-?[\d.]+) p(-?[\d.]+) y(-?[\d.]+)', content)
        if rotation_matches:
            # Swap the order to [yaw, pitch, roll]
            rotation_left = [
                float(rotation_matches[0][2]),  # Yaw
                float(rotation_matches[0][1]),  # Pitch
                float(rotation_matches[0][0])   # Roll
            ]
            rotation_right = [
                float(rotation_matches[1][2]),  # Yaw
                float(rotation_matches[1][1]),  # Pitch
                float(rotation_matches[1][0])   # Roll
            ]
            profile_data['rotation_left'] = rotation_left
            profile_data['rotation_right'] = rotation_right
        else:
            profile_data['rotation_left'] = [0, 0, 0]
            profile_data['rotation_right'] = [0, 0, 0]
        
        # Extract panorama dimensions
        pano_size_match = re.search(r'p f\d+ w(\d+) h(\d+)', content)
        if pano_size_match:
            profile_data['panorama_width'] = int(pano_size_match.group(1))
            profile_data['panorama_height'] = int(pano_size_match.group(2))
        else:
            # Default dimensions if not specified
            profile_data['panorama_width'] = 7776
            profile_data['panorama_height'] = 3888
        
        # Set FOV to 179 degrees to avoid issues at exactly 180 degrees
        profile_data['fov_left'] = 179.0
        profile_data['fov_right'] = 179.0
        
        # Assume zero distortion for simplicity
        profile_data['distortion_left'] = [0, 0, 0, 0, 0]
        profile_data['distortion_right'] = [0, 0, 0, 0, 0]
    
    except FileNotFoundError:
        logger.error(f"PTO profile file not found: {profile_path}")
        raise
    except Exception as e:
        logger.error(f"Failed to load PTO profile: {e}")
        raise
    
    return profile_data

def fisheye_to_equirectangular(fisheye_img, width, height, fov, distortion_params, rotation, debug=False):
    logger.info(f"Converting fisheye to equirectangular with rotation: width={width}, height={height}, fov={fov}, rotation={rotation}")
    
    debug_info = {}

    # Ensure input image is float32
    fisheye_img = np.float32(fisheye_img)
    
    # Fisheye image dimensions
    h_fisheye, w_fisheye = fisheye_img.shape[:2]
    cx, cy = w_fisheye / 2.0, h_fisheye / 2.0  # Center of the fisheye image

    # Generate grid for equirectangular image
    theta = np.linspace(-np.pi, np.pi, width, dtype=np.float32)
    phi = np.linspace(np.pi / 2, -np.pi / 2, height, dtype=np.float32)  # From top to bottom
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

    # Convert back to spherical coordinates
    r = np.sqrt(x_rot**2 + y_rot**2 + z_rot**2)
    theta_fisheye = np.arctan2(y_rot, x_rot)
    phi_fisheye = np.arccos(z_rot / r)

    # Focal length for equisolid angle projection
    f = w_fisheye / 4.0  # Adjusted focal length
    logger.debug(f"Focal length f: {f}")

    if debug:
        debug_info['Focal length f'] = f

    # Radial distance from center (equisolid angle projection)
    r_fisheye = 2 * f * np.sin(phi_fisheye / 2)

    # Fisheye image coordinates
    x_fisheye = cx + r_fisheye * np.cos(theta_fisheye)
    y_fisheye = cy + r_fisheye * np.sin(theta_fisheye)

    # Reshape back to image shape
    map_x = x_fisheye.reshape((height, width)).astype(np.float32)
    map_y = y_fisheye.reshape((height, width)).astype(np.float32)

    # Mask invalid coordinates
    mask = (map_x >= 0) & (map_x < w_fisheye) & (map_y >= 0) & (map_y < h_fisheye)
    valid_points = np.sum(mask)
    total_points = mask.size
    valid_percentage = (valid_points / total_points) * 100
    logger.debug(f"Valid mapping points: {valid_points} out of {total_points} ({valid_percentage:.2f}%)")

    if debug:
        debug_info['Valid mapping points'] = f"{valid_points} out of {total_points} ({valid_percentage:.2f}%)"

    # Use remap to generate the equirectangular image
    try:
        equirectangular = cv2.remap(
            fisheye_img,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
    except Exception as e:
        logger.error(f"Error during remapping: {e}")
        if debug:
            debug_info['Remap Error'] = str(e)
        return None, debug_info

    # Set invalid regions to black
    equirectangular[~mask] = 0

    if debug:
        # Optionally add map_x and map_y stats
        map_x_stats = f"map_x stats: min={np.min(map_x)}, max={np.max(map_x)}, mean={np.mean(map_x)}"
        map_y_stats = f"map_y stats: min={np.min(map_y)}, max={np.max(map_y)}, mean={np.mean(map_y)}"
        debug_info['map_x stats'] = map_x_stats
        debug_info['map_y stats'] = map_y_stats

        # Convert equirectangular image to uint8 for display
        equirectangular_display = (np.clip(equirectangular, 0, 1) * 255).astype(np.uint8)
        debug_info['Equirectangular Image'] = cv2.cvtColor(equirectangular_display, cv2.COLOR_BGR2RGB)

    return equirectangular, debug_info


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
    left_overlap = equi_left[:, half_width - overlap_width: half_width + overlap_width]
    right_overlap = equi_right[:, :2 * overlap_width]

    # Check dimensions
    if left_overlap.shape != right_overlap.shape:
        logger.error(f"Overlap regions have mismatched shapes: left {left_overlap.shape}, right {right_overlap.shape}")
        return None

    # Create alpha blending mask
    alpha = np.linspace(1, 0, 2 * overlap_width)[np.newaxis, :, np.newaxis]  # Shape (1, 2*overlap_width, 1)
    alpha = np.repeat(alpha, height, axis=0)  # Shape (height, 2*overlap_width, 1)

    # Blend the overlapping regions
    blended_overlap = left_overlap * alpha + right_overlap * (1 - alpha)

    # Place blended overlap into the stitched image
    stitched_equirectangular[:, half_width - overlap_width: half_width + overlap_width] = blended_overlap

    # Ensure output is in [0, 1] range
    stitched_equirectangular = np.clip(stitched_equirectangular, 0, 1)

    return stitched_equirectangular

def stitch_gear360_image(gear360_image, pto_profile, image_info, stitching_method='simple', debug=False):
    import logging
    logger = logging.getLogger(__name__)

    debug_images = {}
    try:
        # Ensure input image is float32 and in [0, 1] range
        gear360_image = np.float32(gear360_image) / 255.0

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
            # Convert images to uint8 for display
            fisheye_left_display = (fisheye_left * 255).astype(np.uint8)
            fisheye_right_display = (fisheye_right * 255).astype(np.uint8)
            debug_images['Fisheye Left'] = cv2.cvtColor(fisheye_left_display, cv2.COLOR_BGR2RGB)
            debug_images['Fisheye Right'] = cv2.cvtColor(fisheye_right_display, cv2.COLOR_BGR2RGB)

        # Use profile data for output dimensions and FOV
        output_width = profile_data['panorama_width']
        output_height = profile_data['panorama_height']
        fov_left = profile_data['fov_left']
        fov_right = profile_data['fov_right']
        logger.info(f"Output dimensions: {output_width}x{output_height}, FOV: Left {fov_left}, Right {fov_right}")

        logger.info("Converting fisheye to equirectangular with rotation")

        # Adjust rotations to correct orientation
        rotation_left = [90, 0, 0]    # Rotate left image by +90 degrees (yaw)
        rotation_right = [-90, 0, 0]  # Rotate right image by -90 degrees (yaw)

        # Generate equirectangular images
        equi_left, debug_info_left = fisheye_to_equirectangular(
            fisheye_left,
            output_width // 2,
            output_height,
            fov_left,
            profile_data['distortion_left'],
            rotation_left,
            debug=debug
        )

        equi_right, debug_info_right = fisheye_to_equirectangular(
            fisheye_right,
            output_width // 2,
            output_height,
            fov_right,
            profile_data['distortion_right'],
            rotation_right,
            debug=debug
        )

        if equi_left is None or equi_right is None:
            logger.error("Failed to generate equirectangular images.")
            if debug:
                debug_images['Error'] = "Failed to generate equirectangular images."
                debug_images.update(debug_info_left)
                debug_images.update(debug_info_right)
            return None, debug_images

        logger.debug(f"Equirectangular conversion complete. Dimensions: {equi_left.shape}, {equi_right.shape}")

        if debug:
            # Convert equirectangular images to uint8 for display
            equi_left_display = (np.clip(equi_left, 0, 1) * 255).astype(np.uint8)
            equi_right_display = (np.clip(equi_right, 0, 1) * 255).astype(np.uint8)
            debug_images['Equirectangular Left'] = cv2.cvtColor(equi_left_display, cv2.COLOR_BGR2RGB)
            debug_images['Equirectangular Right'] = cv2.cvtColor(equi_right_display, cv2.COLOR_BGR2RGB)

            # Include debug info
            debug_images.update({f"Left {k}": v for k, v in debug_info_left.items()})
            debug_images.update({f"Right {k}": v for k, v in debug_info_right.items()})

        # Stitch the images
        logger.info(f"Stitching equirectangular images using {stitching_method} method")
        stitched_equirectangular = stitch_simple(equi_left, equi_right, output_width, output_height)

        if stitched_equirectangular is None:
            logger.warning("Stitching failed, returning the left equirectangular image")
            return (equi_left * 255).astype(np.uint8), debug_images

        # Convert back to uint8 for display/saving
        result = (np.clip(stitched_equirectangular, 0, 1) * 255).astype(np.uint8)

        if debug:
            debug_images['Final Stitched Image'] = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        return result, debug_images

    except Exception as e:
        logger.error(f"An error occurred during stitching: {str(e)}", exc_info=True)
        if debug:
            debug_images['Stitching Error'] = str(e)
        return None, debug_images

