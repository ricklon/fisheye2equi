import os
import re
import numpy as np
import cv2
import logging
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)

def log_array_info(name, arr):
    logger.debug(f"{name}: shape={arr.shape}, dtype={arr.dtype}, min={np.min(arr)}, max={np.max(arr)}")



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
        
        # Extract rotation and translation parameters
        rotation_matches = re.findall(r'r([\d.-]+) p([\d.-]+) y([\d.-]+)', content)
        if rotation_matches:
            profile_data['rotation_left'] = [float(x) for x in rotation_matches[0]]
            profile_data['rotation_right'] = [float(x) for x in rotation_matches[1]]
        
        translation_matches = re.findall(r'TrX([\d.-]+) TrY([\d.-]+) TrZ([\d.-]+)', content)
        if translation_matches:
            profile_data['translation_left'] = [float(x) for x in translation_matches[0]]
            profile_data['translation_right'] = [float(x) for x in translation_matches[1]]
        
        logger.debug(f"Loaded rotation values: Left {profile_data['rotation_left']}, Right {profile_data['rotation_right']}")
        logger.debug(f"Loaded translation values: Left {profile_data['translation_left']}, Right {profile_data['translation_right']}")
        
        return profile_data

def fisheye_to_equirectangular(fisheye_img, width, height, fov, distortion_params):
    logger.info(f"Converting fisheye to equirectangular: width={width}, height={height}, fov={fov}")
    
    # Ensure input image is float32 and normalized to [0, 1]
    fisheye_img = np.float32(fisheye_img) / 255.0
    
    # Create camera matrix
    f = width / (2 * np.tan(np.radians(fov / 2)))
    cx, cy = width // 2, height // 2
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)
    
    # Create distortion coefficients array
    D = np.array(distortion_params[:4], dtype=np.float32)
    
    logger.debug(f"Camera matrix K:\n{K}")
    logger.debug(f"Distortion coefficients D: {D}")
    
    # Create equirectangular projection
    equirectangular = np.zeros((height, width, 3), dtype=np.float32)
    
    for y in range(height):
        for x in range(width):
            # Convert equirectangular coordinates to spherical
            theta = (x / width - 0.5) * 2 * np.pi
            phi = (y / height - 0.5) * np.pi
            
            # Convert spherical to 3D Cartesian
            X = np.sin(phi) * np.cos(theta)
            Y = np.sin(phi) * np.sin(theta)
            Z = np.cos(phi)
            
            # Project 3D point to 2D fisheye image plane
            r = np.sqrt(X*X + Y*Y)
            if r == 0:
                continue
            
            theta = np.arctan2(Y, X)
            rho = f * np.arctan2(r, Z)
            
            x_fisheye = cx + rho * np.cos(theta)
            y_fisheye = cy + rho * np.sin(theta)
            
            # Apply fisheye distortion
            x_distorted, y_distorted = cv2.fisheye.distortPoints(np.array([[[x_fisheye, y_fisheye]]], dtype=np.float32), K, D).squeeze()
            
            # Interpolate color from fisheye image
            if 0 <= x_distorted < fisheye_img.shape[1] and 0 <= y_distorted < fisheye_img.shape[0]:
                equirectangular[y, x] = cv2.remap(fisheye_img, np.array([[x_distorted]]), np.array([[y_distorted]]), cv2.INTER_LINEAR)
    
    # Normalize the output
    equirectangular = cv2.normalize(equirectangular, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    logger.debug(f"Equirectangular image shape: {equirectangular.shape}")
    
    return equirectangular

# Rotation methods
def rotate_simple(img, rotation, translation, is_right=False):
    logger.debug(f"Applying simple rotation {rotation} and translation {translation}")
    
    # Ensure input image is float32 and in [0, 1] range
    img = np.clip(np.float32(img), 0, 1)
    
    # Get image dimensions
    height, width = img.shape[:2]
    
    # Create the rotation matrix
    M = cv2.getRotationMatrix2D((width // 2, height // 2), rotation[2], 1.0)
    
    # Apply translation
    M[0, 2] += translation[0]
    M[1, 2] += translation[1]
    
    # Apply transformation
    transformed_img = cv2.warpAffine(img, M, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    # Ensure output is in [0, 1] range
    transformed_img = np.clip(transformed_img, 0, 1)
    
    return transformed_img

def rotate_advanced(img, rotation, translation, is_right=False):
    logger.debug(f"Applying advanced rotation {rotation} and translation {translation}")
    
    # Ensure input image is float32 and in [0, 1] range
    img = np.clip(np.float32(img), 0, 1)
    
    # Convert rotation angles to radians
    rx, ry, rz = np.radians(rotation)
    
    # Create rotation matrices
    Rx = np.array([[1, 0, 0], 
                   [0, np.cos(rx), -np.sin(rx)], 
                   [0, np.sin(rx), np.cos(rx)]], dtype=np.float32)
    
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], 
                   [0, 1, 0], 
                   [-np.sin(ry), 0, np.cos(ry)]], dtype=np.float32)
    
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], 
                   [np.sin(rz), np.cos(rz), 0], 
                   [0, 0, 1]], dtype=np.float32)
    
    # Combine rotation matrices
    R = Rz @ Ry @ Rx
    
    # Get image dimensions
    height, width = img.shape[:2]
    
    # Create 3x3 transformation matrix
    M = np.eye(3, dtype=np.float32)
    M[:2, :2] = R[:2, :2]
    M[:2, 2] = translation[:2]
    
    # Apply transformation
    transformed_img = cv2.warpPerspective(img, M, (width, height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    
    # Ensure output is in [0, 1] range
    transformed_img = np.clip(transformed_img, 0, 1)
    
    return transformed_img


def apply_rotation_translation(img, rotation, translation, is_right=False):
    logger.debug(f"Applying rotation {rotation} and translation {translation} to {'right' if is_right else 'left'} image")
    
    # Ensure input image is float32 and in [0, 1] range
    img = np.clip(np.float32(img), 0, 1)
    
    # Get image dimensions
    height, width = img.shape[:2]
    logger.debug(f"Image dimensions: {width}x{height}")
    
    if is_right:
        # For the right image, apply only a small rotation
        # Convert rotation angles to radians, but use only a fraction of the yaw
        rx, ry, rz = np.radians(rotation)
        rz = rz * 0.01  # Apply only 1% of the yaw rotation, adjust as needed
    else:
        # For the left image, no rotation
        rx, ry, rz = 0, 0, 0
    
    logger.debug(f"Adjusted rotation angles in radians: rx={rx}, ry={ry}, rz={rz}")
    
    # Create the rotation matrix
    M = cv2.getRotationMatrix2D((width // 2, height // 2), np.degrees(rz), 1.0)
    
    # Apply translation
    M[0, 2] += translation[0]
    M[1, 2] += translation[1]
    
    logger.debug(f"Transformation matrix M:\n{M}")
    
    # Apply transformation
    transformed_img = cv2.warpAffine(img, M, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    # Save debug image
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(img)
    plt.title("Before rotation")
    plt.subplot(122)
    plt.imshow(transformed_img)
    plt.title(f"After rotation - {'Right' if is_right else 'Left'}")
    plt.savefig(f"rotation_debug_{'right' if is_right else 'left'}_{rotation[0]}_{rotation[1]}_{rotation[2]}.png")
    plt.close()
    
    # Ensure output is in [0, 1] range
    transformed_img = np.clip(transformed_img, 0, 1)
    
    return transformed_img


def find_match_loc(ref_img, tmpl_img, method=cv2.TM_CCOEFF_NORMED):
    if ref_img is None or tmpl_img is None:
        return None

    result = cv2.matchTemplate(ref_img, tmpl_img, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    match_loc = max_loc if method not in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED] else min_loc
    return match_loc

def create_control_points(match_loc_left, match_loc_right, equi_left, equi_right, template_size=(200, 100), num_points=10):
    if match_loc_left is None or match_loc_right is None:
        return None, None

    h, w = template_size
    half_h, half_w = h // 2, w // 2

    start_y = max(match_loc_left[1] - half_h, 0)
    end_y = min(match_loc_left[1] + half_h, equi_left.shape[0])
    start_x_left = max(match_loc_left[0] - half_w, 0)
    end_x_left = min(match_loc_left[0] + half_w, equi_left.shape[1])
    start_x_right = max(match_loc_right[0] - half_w, 0)
    end_x_right = min(match_loc_right[0] + half_w, equi_right.shape[1])

    control_points_left = []
    control_points_right = []

    for _ in range(num_points):
        y = np.random.randint(start_y, end_y)
        x_left = np.random.randint(start_x_left, end_x_left)
        x_right = np.random.randint(start_x_right, end_x_right)

        control_points_left.append([x_left, y])
        control_points_right.append([x_right, y])

    return np.array(control_points_left, dtype=np.float32), np.array(control_points_right, dtype=np.float32)

def stitch_equirectangular_pair(equi_left, equi_right, width, height, fov):
    logger.info(f"Stitching equirectangular pair: width={width}, height={height}, fov={fov}")
    
    try:
        # Ensure input images are float32 and in [0, 1] range
        equi_left = np.clip(np.float32(equi_left), 0, 1)
        equi_right = np.clip(np.float32(equi_right), 0, 1)
        
        log_array_info("equi_left", equi_left)
        log_array_info("equi_right", equi_right)
        
        # Resize images if they don't match the expected dimensions
        expected_width = width // 2
        if equi_left.shape[1] != expected_width or equi_right.shape[1] != expected_width:
            logger.warning(f"Resizing images to match expected width of {expected_width}")
            equi_left = cv2.resize(equi_left, (expected_width, height))
            equi_right = cv2.resize(equi_right, (expected_width, height))
        
        # Create a mask for blending
        blend_width = min(width // 8, equi_left.shape[1] // 4)  # Ensure blend width isn't too large
        mask = np.zeros((height, expected_width), dtype=np.float32)
        mask[:, -blend_width:] = np.tile(np.linspace(1, 0, blend_width, dtype=np.float32), (height, 1))

        log_array_info("mask", mask)

        # Initialize the stitched image
        stitched_equirectangular = np.zeros((height, width, 3), dtype=np.float32)
        
        for c in range(3):  # Blend each color channel separately
            # Copy left image
            stitched_equirectangular[:, :expected_width, c] = equi_left[:, :, c]
            
            # Blend the overlapping region
            blend_left = equi_left[:, -blend_width:, c]
            blend_right = equi_right[:, :blend_width, c]
            blended = blend_left * mask[:, -blend_width:] + blend_right * (1 - mask[:, -blend_width:])
            
            # Copy blended region
            stitched_equirectangular[:, expected_width-blend_width:expected_width, c] = blended
            
            # Copy right image
            right_width = min(expected_width, equi_right.shape[1] - blend_width)
            stitched_equirectangular[:, expected_width:expected_width+right_width, c] = equi_right[:, blend_width:blend_width+right_width, c]
        
        # Ensure output is in [0, 1] range
        stitched_equirectangular = np.clip(stitched_equirectangular, 0, 1)
        
        log_array_info("stitched_equirectangular", stitched_equirectangular)
        
        logger.info("Stitching completed successfully")
        return stitched_equirectangular

    except Exception as e:
        logger.error(f"An error occurred during stitching: {str(e)}", exc_info=True)
        # Fallback: return a padded version of the left image if stitching fails
        logger.warning("Falling back to returning a padded version of the left image")
        padded_left = np.zeros((height, width, 3), dtype=np.float32)
        padded_left[:, :equi_left.shape[1], :] = equi_left
        return padded_left
    



# Stitching methods
def stitch_simple(equi_left, equi_right, width, height):
    logger.info(f"Stitching equirectangular pair using simple method: width={width}, height={height}")
    
    # Ensure input images are float32 and in [0, 1] range
    equi_left = np.clip(np.float32(equi_left), 0, 1)
    equi_right = np.clip(np.float32(equi_right), 0, 1)
    
    # Create a mask for blending
    blend_width = width // 8
    mask = np.zeros((height, width), dtype=np.float32)
    mask[:, width//2-blend_width:width//2+blend_width] = np.tile(
        np.linspace(1, 0, 2*blend_width, dtype=np.float32), (height, 1)
    )
    
    # Perform blending
    stitched_equirectangular = np.zeros((height, width, 3), dtype=np.float32)
    
    for c in range(3):  # Blend each color channel separately
        stitched_equirectangular[:, :width//2, c] = equi_left[:, :, c]
        stitched_equirectangular[:, width//2:, c] = equi_right[:, :, c]
        stitched_equirectangular[:, width//2-blend_width:width//2+blend_width, c] = cv2.addWeighted(
            equi_left[:, -blend_width:, c], mask[:, width//2-blend_width:width//2+blend_width],
            equi_right[:, :blend_width, c], 1 - mask[:, width//2-blend_width:width//2+blend_width],
            0
        )
    
    # Ensure output is in [0, 1] range
    stitched_equirectangular = np.clip(stitched_equirectangular, 0, 1)
    
    return stitched_equirectangular

def stitch_advanced(equi_left, equi_right, width, height):
    logger.info(f"Stitching equirectangular pair using advanced method: width={width}, height={height}")
    
    # Ensure input images are float32 and in [0, 1] range
    equi_left = np.clip(np.float32(equi_left), 0, 1)
    equi_right = np.clip(np.float32(equi_right), 0, 1)
    
    # Find matching features
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(equi_left, None)
    kp2, des2 = orb.detectAndCompute(equi_right, None)
    
    # Match features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Find homography
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    # Warp right image
    equi_right_aligned = cv2.warpPerspective(equi_right, M, (width, height))
    
    # Blend images
    blend_width = width // 8
    mask = np.zeros((height, width), dtype=np.float32)
    mask[:, width//2-blend_width:width//2+blend_width] = np.tile(
        np.linspace(1, 0, 2*blend_width, dtype=np.float32), (height, 1)
    )
    
    stitched_equirectangular = np.zeros((height, width, 3), dtype=np.float32)
    
    for c in range(3):
        stitched_equirectangular[:, :width//2, c] = equi_left[:, :, c]
        stitched_equirectangular[:, width//2:, c] = equi_right_aligned[:, width//2:, c]
        stitched_equirectangular[:, width//2-blend_width:width//2+blend_width, c] = cv2.addWeighted(
            equi_left[:, -blend_width:, c], mask[:, width//2-blend_width:width//2+blend_width],
            equi_right_aligned[:, :blend_width, c], 1 - mask[:, width//2-blend_width:width//2+blend_width],
            0
        )
    
    # Ensure output is in [0, 1] range
    stitched_equirectangular = np.clip(stitched_equirectangular, 0, 1)
    
    return stitched_equirectangular



def stitch_gear360_image(gear360_image, pto_profile, image_info, rotation_method='simple', stitching_method='simple', debug=False):
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

        # Apply rotation to left and right images separately
        logger.info(f"Applying {rotation_method} rotation to left and right images")
        rotate_func = rotate_simple if rotation_method == 'simple' else rotate_advanced
        rotated_left = rotate_func(fisheye_left, profile_data['rotation_left'], profile_data['translation_left'], is_right=False)
        rotated_right = rotate_func(fisheye_right, profile_data['rotation_right'], profile_data['translation_right'], is_right=True)

        if debug:
            debug_images['Rotated Left'] = (rotated_left * 255).astype(np.uint8)
            debug_images['Rotated Right'] = (rotated_right * 255).astype(np.uint8)

        # Use profile data for output dimensions and FOV
        output_width = profile_data['panorama_width']
        output_height = profile_data['panorama_height']
        fov_left = profile_data['fov_left']
        fov_right = profile_data['fov_right']
        logger.info(f"Output dimensions: {output_width}x{output_height}, FOV: Left {fov_left}, Right {fov_right}")

        logger.info("Converting fisheye to equirectangular")
        equi_left = fisheye_to_equirectangular(rotated_left, output_width // 2, output_height, fov_left, profile_data['distortion_left'])
        equi_right = fisheye_to_equirectangular(rotated_right, output_width // 2, output_height, fov_right, profile_data['distortion_right'])
        logger.debug(f"Equirectangular conversion complete. Dimensions: {equi_left.shape}, {equi_right.shape}")

        if debug:
            debug_images['Equirectangular Left'] = (equi_left * 255).astype(np.uint8)
            debug_images['Equirectangular Right'] = (equi_right * 255).astype(np.uint8)

        logger.info(f"Stitching equirectangular images using {stitching_method} method")
        stitch_func = stitch_simple if stitching_method == 'simple' else stitch_advanced
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


    def visualize_rotation(original, rotated, title):
        plt.figure(figsize=(20, 10))
        plt.subplot(121)
        plt.imshow(original)
        plt.title("Original")
        plt.subplot(122)
        plt.imshow(rotated)
        plt.title(f"Rotated - {title}")
        plt.savefig(f"rotation_debug_{title}.png")
        plt.close()

# If you want to add any utility functions or constants, you can add them here