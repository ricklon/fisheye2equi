import numpy as np
import cv2
import streamlit as st

# Constants for polynomial coefficients
P1 = -7.5625e-17
P2 = 1.9589e-13
P3 = -1.8547e-10
P4 = 6.1997e-08
P5 = -6.9432e-05
P6 = 0.9976

def fish2Eqt(x_dest, y_dest, W_rad):
    phi = x_dest / W_rad
    theta = -y_dest / W_rad + np.pi / 2

    if theta < 0:
        theta = -theta
        phi += np.pi
    if theta > np.pi:
        theta = np.pi - (theta - np.pi)
        phi += np.pi

    s = np.sin(theta)
    v = np.array([s * np.sin(phi), np.cos(theta), s * np.cos(phi)])
    r = np.sqrt(v[1] * v[1] + v[0] * v[0])
    theta = W_rad * np.arctan2(r, v[2])

    x_src = theta * v[0] / r
    y_src = theta * v[1] / r

    return x_src, y_src

def fish2Map(width, height, fov):
    map_x = np.zeros((height, width), dtype=np.float32)
    map_y = np.zeros((height, width), dtype=np.float32)
    W_rad = width / (2.0 * np.pi)
    w2 = width / 2 - 0.5
    h2 = height / 2 - 0.5

    for y in range(height):
        y_d = y - h2
        for x in range(width):
            x_d = x - w2
            x_s, y_s = fish2Eqt(x_d, y_d, W_rad)
            map_x[y, x] = x_s + w2
            map_y[y, x] = y_s + h2

    return map_x, map_y


def blend_left(bg1, bg2):
    alpha = np.tile(np.linspace(1, 0, bg1.shape[1]), (bg1.shape[0], 1)).reshape(bg1.shape[0], bg1.shape[1], 1)
    return (alpha * bg1 + (1 - alpha) * bg2).astype(np.uint8)

def blend_right(bg1, bg2):
    alpha = np.tile(np.linspace(0, 1, bg1.shape[1]), (bg1.shape[0], 1)).reshape(bg1.shape[0], bg1.shape[1], 1)
    return (alpha * bg1 + (1 - alpha) * bg2).astype(np.uint8)



def blend(left_img, right_img_aligned, crop, p_x1, p_x2, p_wid, row_start, row_end):
    h, w = left_img.shape[:2]
    sideW = 45  # Adjust this value based on your requirements

    left_img_cr = left_img[:, w//2-crop:w//2+crop]

    for r in range(row_start, row_end):
        # Left boundary
        lf_win_1 = left_img_cr[r, p_x1-sideW:p_x1+sideW]
        rt_win_1 = right_img_aligned[r, p_x1-sideW:p_x1+sideW]

        # Right boundary
        lf_win_2 = left_img_cr[r, w-p_x2-sideW:w-p_x2+sideW]
        rt_win_2 = right_img_aligned[r, w-p_x2-sideW:w-p_x2+sideW]

        # Blend (ramp)
        bleft = blend_left(lf_win_1, rt_win_1)
        bright = blend_right(lf_win_2, rt_win_2)

        # Update left boundary
        left_img_cr[r, p_x1-sideW:p_x1+sideW] = bleft
        right_img_aligned[r, p_x1-sideW:p_x1+sideW] = bleft

        # Update right boundary
        left_img_cr[r, w-p_x2-sideW:w-p_x2+sideW] = bright
        right_img_aligned[r, w-p_x2-sideW:w-p_x2+sideW] = bright

    # Combine the left and right images
    result = np.zeros((h, w*2, 3), dtype=np.uint8)
    result[:, :w//2-crop] = left_img[:, :w//2-crop]
    result[:, w//2-crop:w//2+crop] = left_img_cr
    result[:, w//2+crop:] = left_img[:, w//2+crop:]
    result[:, w-crop:w+crop] = cv2.bitwise_or(result[:, w-crop:w+crop], right_img_aligned)

    return result

def fisheye_to_equirectangular(fisheye_img, width, height, fov=193):
    map_x, map_y = fish2Map(width, height, fov)
    equirectangular = cv2.remap(fisheye_img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return equirectangular

def compensate_light(equirectangular_img, scale_map):
    if equirectangular_img is None:
        return None

    # Split the image into BGR channels
    b, g, r = cv2.split(equirectangular_img)

    # Apply the scale map to each channel
    b_compensated = cv2.multiply(b.astype(np.float32), scale_map)
    g_compensated = cv2.multiply(g.astype(np.float32), scale_map)
    r_compensated = cv2.multiply(r.astype(np.float32), scale_map)

    # Merge the compensated channels back into an image
    compensated_img = cv2.merge((b_compensated, g_compensated, r_compensated))
    compensated_img = np.clip(compensated_img, 0, 255).astype(np.uint8)

    return compensated_img

def create_circular_mask(h, w, fov):
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    r = int(min(cx, cy) * fov / 360)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    return mask

def calculate_overlap(width, fov):
    max_fov = 195.0  # Adjust this value based on the maximum field of view of the camera
    crop = int(0.5 * width * (max_fov - 180.0) / max_fov)
    return crop

def find_match_loc(ref_img, tmpl_img, method=cv2.TM_CCOEFF_NORMED):
    if ref_img is None or tmpl_img is None:
        return None

    result = cv2.matchTemplate(ref_img, tmpl_img, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    match_loc = max_loc if method not in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED] else min_loc
    if match_loc and len(match_loc) == 2:
        return match_loc
    else:
        return None


def validate_match_locations(match_loc_left, match_loc_right, equi_left, equi_right):
    # Check if match locations are valid
    if match_loc_left is None or match_loc_right is None:
        print("Error: Match locations are None.")
        return None, None

    # Convert match locations to integers and ensure they are 1D arrays with size 2
    match_loc_left = np.array(match_loc_left, dtype=int)
    match_loc_right = np.array(match_loc_right, dtype=int)
    
    if match_loc_left.size != 2 or match_loc_right.size != 2:
        print("Error: Match locations should be arrays of size 2.")
        return None, None
    
    return match_loc_left, match_loc_right

def generate_control_points(match_loc_left, match_loc_right, equi_left, equi_right, template_size=(200, 100), num_points=10):
    h, w = template_size
    half_h, half_w = h // 2, w // 2
    
    # Calculate bounds to ensure the sampling area is within the image dimensions
    bounds_left = [
        max(match_loc_left[1] - half_h, 0),
        min(match_loc_left[1] + half_h, equi_left.shape[0]),
        max(match_loc_left[0] - half_w, 0),
        min(match_loc_left[0] + half_w, equi_left.shape[1])
    ]
    
    bounds_right = [
        max(match_loc_right[1] - half_h, 0),
        min(match_loc_right[1] + half_h, equi_right.shape[0]),
        max(match_loc_right[0] - half_w, 0),
        min(match_loc_right[0] + half_w, equi_right.shape[1])
    ]

    # Generate random control points within the defined bounds
    ys = np.random.randint(bounds_left[0], bounds_left[1], size=num_points)
    xs_left = np.random.randint(bounds_left[2], bounds_left[3], size=num_points)
    xs_right = np.random.randint(bounds_right[2], bounds_right[3], size=num_points)

    control_points_left = np.column_stack((xs_left, ys))
    control_points_right = np.column_stack((xs_right, ys))

    return control_points_left.astype(np.float32), control_points_right.astype(np.float32)


def create_control_points(match_loc_left, match_loc_right, equi_left, equi_right, template_size=(200, 100), num_points=10):
    if match_loc_left is None or match_loc_right is None:
        return None, None

    # Convert match locations to integer if they are arrays
    match_loc_left = np.array(match_loc_left, dtype=int)
    match_loc_right = np.array(match_loc_right, dtype=int)

    # Ensure they are not empty and are 1D arrays with size 2
    if match_loc_left.size != 2 or match_loc_right.size != 2:
        print("Error: Match locations should be arrays of size 2.")
        return None, None

    h, w = template_size
    half_h, half_w = h // 2, w // 2

    # Ensure the use of integer indices
    start_y = max(match_loc_left[1] - half_h, 0)
    end_y = min(match_loc_left[1] + half_h, equi_left.shape[0])
    start_x_left = max(match_loc_left[0] - half_w, 0)
    end_x_left = min(match_loc_left[0] + half_w, equi_left.shape[1])
    start_x_right = max(match_loc_right[0] - half_w, 0)
    end_x_right = min(match_loc_right[0] + half_w, equi_right.shape[1])

    # Generate control points
    control_points_left = []
    control_points_right = []

    for _ in range(num_points):
        y = np.random.randint(start_y, end_y)
        x_left = np.random.randint(start_x_left, end_x_left)
        x_right = np.random.randint(start_x_right, end_x_right)

        control_points_left.append([x_left, y])
        control_points_right.append([x_right, y])

    control_points_left = np.array(control_points_left, dtype=np.float32)
    control_points_right = np.array(control_points_right, dtype=np.float32)

    return control_points_left, control_points_right

def create_scale_map(height, width):
    # Create a scale map based on the reverse light fall-off profile
    # Adjust the values based on your camera specifications
    scale_map = np.ones((height, width), dtype=np.float32)
    cx, cy = width // 2, height // 2
    for y in range(height):
        for x in range(width):
            r = np.sqrt((x - cx)**2 + (y - cy)**2)
            scale_map[y, x] = 1 / (1 + 0.001 * r)
    return scale_map

def stitch_equirectangular_pair(equi_left, equi_right, width, height, fov, match_loc_left, match_loc_right, template_size=(200, 100), num_points=10):
    if equi_left is None or equi_right is None:
        return None

    control_points_left, control_points_right = create_control_points(match_loc_left, match_loc_right, equi_left, equi_right, template_size, num_points)
    if control_points_left is None or control_points_right is None:
        print("Control points could not be generated.")
        return None

    # Calculate the overlap region
    crop = calculate_overlap(width // 2, fov)
    p_x1_ref = 2 * crop
    p_x2_ref = width // 2 - 2 * crop + 1

    # Create blending patches
    row_start = height // 4
    row_end = 3 * height // 4
    p_wid = 55  # Adjust this value based on your requirements
    p_x1 = 90 - 15  # Adjust this value based on your requirements
    p_x2 = width // 2 - p_x1

    if control_points_left is not None and control_points_right is not None:
        if len(control_points_left) != len(control_points_right):
            print("Error: Control points arrays must have the same number of points.")
            return None
        # Ensure the control points are numpy arrays of the correct shape and type
        control_points_left = np.array(control_points_left, dtype=np.float32).reshape(-1, 1, 2)
        control_points_right = np.array(control_points_right, dtype=np.float32).reshape(-1, 1, 2)

        # Find the homography matrix using RANSAC
        homography, status = cv2.findHomography(control_points_right, control_points_left, cv2.RANSAC)
        if homography is None:
            print("Homography could not be calculated.")
            return None

        # Warp the right image using the homography matrix
        equi_right_aligned = cv2.warpPerspective(equi_right, homography, (width, height))
    else:
        # If control points are not provided, resize the right image to match the left image
        equi_right_aligned = cv2.resize(equi_right, (equi_left.shape[1], equi_left.shape[0]))

    # Perform blending
    stitched_equirectangular = blend(equi_left, equi_right_aligned, crop, p_x1, p_x2, p_wid, row_start, row_end)

    return stitched_equirectangular

def resize_image(image, max_width=800):
    h, w = image.shape[:2]
    if w > max_width:
        ratio = max_width / w
        new_height = int(h * ratio)
        resized_image = cv2.resize(image, (max_width, new_height))
    else:
        resized_image = image.copy()
    return resized_image

def main():
    st.title("Gear 360 Fisheye to Equirectangular Conversion")

    # File upload
    uploaded_file = st.file_uploader("Choose a Gear 360 image file", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Read and display the image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        gear360_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        gear360_image_rgb = cv2.cvtColor(gear360_image, cv2.COLOR_BGR2RGB)
        resized_image = resize_image(gear360_image_rgb)
        st.image(resized_image, caption="Uploaded Gear 360 Image", use_column_width=True)

        # Checkbox for resizing images for testing
        resize_for_testing = st.checkbox("Resize Images for Testing", value=True)
        
        if resize_for_testing:
            test_size = (640, 320)
            gear360_image_resized = cv2.resize(gear360_image, test_size)
            fisheye_left = gear360_image_resized[:, :test_size[0] // 2]
            fisheye_right = gear360_image_resized[:, test_size[0] // 2:]
            output_width = test_size[0]
            output_height = test_size[1] // 2
        else:
            h, w = gear360_image.shape[:2]
            fisheye_left = gear360_image[:, :w // 2]
            fisheye_right = gear360_image[:, w // 2:]
            output_width = st.number_input("Output Width", value=4096, min_value=1024, max_value=8192, step=1)
            output_height = output_width // 2

        fov = st.number_input("Field of View (FOV)", value=193, min_value=180, max_value=220, step=1)
        equi_left = fisheye_to_equirectangular(fisheye_left, output_width // 2, output_height, fov)
        equi_right = fisheye_to_equirectangular(fisheye_right, output_width // 2, output_height, fov)

        st.subheader("Preview")
        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(equi_left, cv2.COLOR_BGR2RGB), caption="Left Equirectangular Image", use_column_width=True)
        with col2:
            st.image(cv2.cvtColor(equi_right, cv2.COLOR_BGR2RGB), caption="Right Equirectangular Image", use_column_width=True)

        

        # Stitch button
        if st.button("Stitch Images"):
            # Find matching locations
            match_loc_left = find_match_loc(equi_left, equi_right)
            match_loc_right = find_match_loc(equi_right, equi_left)

            # Validate and generate control points
            validated_left, validated_right = validate_match_locations(match_loc_left, match_loc_right, equi_left, equi_right)

            if validated_left is not None and validated_right is not None:
                control_points_left, control_points_right = generate_control_points(validated_left, validated_right, equi_left, equi_right)
                
                # Perform the stitching
                stitched_equirectangular = stitch_equirectangular_pair(equi_left, equi_right, output_width, output_height, fov, control_points_left, control_points_right)
                if stitched_equirectangular is not None:
                    st.subheader("Stitched Image")
                    # Checkbox for user to choose between original or resized image
                    show_resized = st.checkbox("Show Resized Image", value=True)
                    if show_resized:
                        resized_stitched = resize_image(cv2.cvtColor(stitched_equirectangular, cv2.COLOR_BGR2RGB))
                        st.image(resized_stitched, caption="Resized Stitched Equirectangular Image", use_column_width=True)
                    else:
                        st.image(cv2.cvtColor(stitched_equirectangular, cv2.COLOR_BGR2RGB), caption="Original Stitched Equirectangular Image", use_column_width=True)
                else:
                    st.error("Failed to stitch the equirectangular images.")
            else:
                st.error("Control point validation failed.")

            # Save stitched image
            if stitched_equirectangular is not None and st.button("Save Equirectangular Image"):
                cv2.imwrite("stitched_equirectangular.png", stitched_equirectangular)
                st.success("Equirectangular image saved as 'stitched_equirectangular.png'.")

if __name__ == "__main__":
    main()