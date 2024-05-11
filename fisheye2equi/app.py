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

def fisheye_to_equirectangular(fisheye_img, width, height, fov=193):
    equirectangular = np.zeros((height, width, 3), dtype=np.uint8)
    if fisheye_img is None:
        return equirectangular, {}

    h, w = fisheye_img.shape[:2]
    cx, cy = w // 2, h // 2  # Center of the fisheye image
    r_max = min(cx, cy)  # Maximum radius of the fisheye

    for y_eq in range(height):
        theta = (y_eq / height - 0.5) * np.pi
        for x_eq in range(width):
            phi = (0.5 - x_eq / width) * 2 * np.pi

            # Print theta and phi for a few sample points
            if x_eq % 100 == 0 and y_eq % 100 == 0:
                print(f"Sample point: (x_eq={x_eq}, y_eq={y_eq})")
                print(f"  Theta: {theta:.4f}")
                print(f"  Phi: {phi:.4f}")

            # Calculate fisheye radius using polynomial coefficients
            r = P1 * phi**5 + P2 * phi**4 + P3 * phi**3 + P4 * phi**2 + P5 * phi + P6
            r *= r_max

            # Calculate fisheye coordinates
            x_fish = int(cx + r * np.cos(theta))
            y_fish = int(cy + r * np.sin(theta))

            if 0 <= x_fish < w and 0 <= y_fish < h:
                equirectangular[y_eq, x_eq, :] = fisheye_img[y_fish, x_fish, :]

    # Debugging: Collect debugging information
    debug_info = {
        "x_fish range": (np.min(x_fish), np.max(x_fish)),
        "y_fish range": (np.min(y_fish), np.max(y_fish)),
        "Equirectangular dimensions": equirectangular.shape,
        "Number of non-zero pixels": np.sum(equirectangular > 0)
    }

    return equirectangular, debug_info

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

def create_circular_mask(h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    r = min(cx, cy)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    return mask

def find_match_loc(ref_img, tmpl_img, method=cv2.TM_CCOEFF_NORMED):
    if ref_img is None or tmpl_img is None:
        return None

    # Perform template matching
    result = cv2.matchTemplate(ref_img, tmpl_img, method)

    # Find the location of the best match
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        # For methods based on squared differences, the best match is the minimum value
        match_loc = min_loc
    else:
        # For other methods, the best match is the maximum value
        match_loc = max_loc

    return match_loc

def create_control_points(match_loc_left, match_loc_right, equi_left, equi_right, template_size=(200, 100), num_points=10):
    if match_loc_left is None or match_loc_right is None:
        return None, None

    h, w = template_size
    half_h, half_w = h // 2, w // 2

    # Calculate the range of coordinates for control points
    start_y = max(match_loc_left[1] - half_h, 0)
    end_y = min(match_loc_left[1] + half_h, equi_left.shape[0])
    start_x_left = max(match_loc_left[0] - half_w, 0)
    end_x_left = min(match_loc_left[0] + half_w, equi_left.shape[1])
    start_x_right = max(match_loc_right[0] - half_w, 0)
    end_x_right = min(match_loc_right[0] + half_w, equi_right.shape[1])

    # Generate control points
    control_points_left = []
    control_points_right = []

    for i in range(num_points):
        y = np.random.randint(start_y, end_y)
        x_left = np.random.randint(start_x_left, end_x_left)
        x_right = np.random.randint(start_x_right, end_x_right)

        control_points_left.append((x_left, y))
        control_points_right.append((x_right, y))

    return control_points_left, control_points_right

def create_scale_map(height, width):
    # Create a scale map based on the reverse light fall-off profile
    # Adjust the values based on your camera specifications
    scale_map = np.ones((height, width), dtype=np.float32)
    cx, cy = width // 2, height // 2
    for y in range(height):
        for x in range(width):
            r = np.sqrt((x - cx)**2 + (y - cy)**2)
            scale_map[y, x] = 1 / (1 + 0.01 * r)
    return scale_map

def stitch_equirectangular_pair(equi_left, equi_right, width, height, control_points_left=None, control_points_right=None, blend_mask=None):
    if equi_left is None or equi_right is None:
        return None

    if control_points_left is not None and control_points_right is not None:
        control_points_left = np.array(control_points_left, dtype=np.float32)
        control_points_right = np.array(control_points_right, dtype=np.float32)
        homography, _ = cv2.findHomography(control_points_right, control_points_left, cv2.RANSAC)
        equi_right_aligned = cv2.warpPerspective(equi_right, homography, (width, height))
    else:
        equi_right_aligned = cv2.resize(equi_right, (equi_left.shape[1], equi_left.shape[0]))

    # Combine images side by side
    combined_image = np.concatenate((equi_left, equi_right_aligned), axis=1)

    # Apply the blending mask
    if blend_mask is not None:
        if combined_image.shape[1] != blend_mask.shape[1]:
            blend_mask = cv2.resize(blend_mask, (combined_image.shape[1], combined_image.shape[0]))

        stitched = combined_image * blend_mask
        stitched = np.clip(stitched, 0, 255).astype(np.uint8)
    else:
        stitched = combined_image

    return stitched

def create_blend_mask(total_width, height, overlap_width):
    mask = np.zeros((height, total_width), dtype=np.float32)
    start_fade = total_width // 2 - overlap_width // 2
    end_fade = total_width // 2 + overlap_width // 2

    # Gradient within the overlap area
    mask[:, :start_fade] = 1
    mask[:, start_fade:end_fade] = np.linspace(1, 0, end_fade - start_fade)
    mask[:, end_fade:] = 0

    # Convert mask to 3 channels
    mask = np.stack([mask]*3, axis=-1)
    return mask

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
        # Read the uploaded image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        gear360_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Convert the uploaded image from BGR to RGB
        gear360_image_rgb = cv2.cvtColor(gear360_image, cv2.COLOR_BGR2RGB)

        # Resize the image for display
        resized_image = resize_image(gear360_image_rgb)

        # Display the resized uploaded image
        st.image(resized_image, caption="Uploaded Gear 360 Image", use_column_width=True)

        # Split the Gear 360 image into left and right fisheye images
        h, w = gear360_image.shape[:2]
        fisheye_left = gear360_image[:, :w // 2, :]
        fisheye_right = gear360_image[:, w // 2:, :]


        # Debugging: Collect fisheye image dimensions
        fisheye_debug_info = {
            "Left fisheye shape": fisheye_left.shape,
            "Right fisheye shape": fisheye_right.shape
        }

        # Input fields for output size and FOV
        output_width = st.number_input("Output Width", value=4096, min_value=1024, max_value=8192, step=1)
        output_height = output_width // 2
        fov = st.number_input("Field of View (FOV)", value=193, min_value=180, max_value=220, step=1)

        # Debugging: Collect FOV value
        fov_debug_info = {
            "Field of View (FOV)": fov
        }


        # Create circular masks
        mask_left = create_circular_mask(h, w // 2)
        mask_right = create_circular_mask(h, w // 2)

        # Apply circular masks to fisheye images
        fisheye_left_masked = cv2.bitwise_and(fisheye_left, fisheye_left, mask=mask_left)
        fisheye_right_masked = cv2.bitwise_and(fisheye_right, fisheye_right, mask=mask_right)

       # Display the masked fisheye images side by side
        st.subheader("Masked Fisheye Images")
        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(fisheye_left_masked, cv2.COLOR_BGR2RGB), caption="Left Fisheye Image (Masked)", use_column_width=True)
        with col2:
            st.image(cv2.cvtColor(fisheye_right_masked, cv2.COLOR_BGR2RGB), caption="Right Fisheye Image (Masked)", use_column_width=True)

        # Display the circular masks
        st.subheader("Circular Masks")
        col1, col2 = st.columns(2)
        with col1:
            st.image(mask_left, caption="Left Mask", use_column_width=True)
        with col2:
            st.image(mask_right, caption="Right Mask", use_column_width=True)



        # Checkbox for enabling light compensation
        enable_light_compensation = st.checkbox("Enable Light Compensation", value=True)

        # Create scale map for light compensation
        scale_map = create_scale_map(output_height, output_width // 2)

        # Checkbox for enabling refined alignment
        enable_refined_alignment = st.checkbox("Enable Refined Alignment", value=True)

        # Slider for adjusting overlap width
        overlap_width = st.slider("Overlap Width", min_value=50, max_value=500, value=200, step=10)

        # Convert fisheye images to equirectangular
        equi_left, debug_info_left = fisheye_to_equirectangular(fisheye_left_masked, output_width // 2, output_height, fov)
        equi_right, debug_info_right = fisheye_to_equirectangular(fisheye_right_masked, output_width // 2, output_height, fov)

        # Debugging: Collect equirectangular image dimensions
        equi_debug_info = {
            "Left equirectangular shape": equi_left.shape,
            "Right equirectangular shape": equi_right.shape
        }

        # Display debugging information to the user
        st.subheader("Debugging Information")
        st.write("Fisheye Image Dimensions:")
        st.info(fisheye_debug_info)
        st.write("Field of View (FOV):")
        st.info(fov_debug_info)
        st.write("Fisheye to Equirectangular Conversion (Left):")
        st.info(debug_info_left)
        st.write("Fisheye to Equirectangular Conversion (Right):")
        st.info(debug_info_right)
        st.write("Equirectangular Image Dimensions:")
        st.info(equi_debug_info)

        if enable_light_compensation:
            equi_left = compensate_light(equi_left, scale_map)
            equi_right = compensate_light(equi_right, scale_map)

        # Display the equirectangular images
        st.subheader("Preview")
        st.image(cv2.cvtColor(equi_left, cv2.COLOR_BGR2RGB), caption="Left Equirectangular Image", use_column_width=True)
        st.image(cv2.cvtColor(equi_right, cv2.COLOR_BGR2RGB), caption="Right Equirectangular Image", use_column_width=True)

        # Stitch button
        if st.button("Stitch Images"):
            # Default initialization of control points
            control_points_left = None
            control_points_right = None

            if enable_refined_alignment:
                # Find matching locations
                match_loc_left = find_match_loc(equi_left, equi_right)
                match_loc_right = find_match_loc(equi_right, equi_left)

                if match_loc_left is not None and match_loc_right is not None:
                    # Generate control points
                    control_points_left, control_points_right = create_control_points(match_loc_left, match_loc_right, equi_left, equi_right)
                else:
                    st.warning("Failed to find reliable matches for control points. Proceeding with simple alignment.")

            # Generate the blend mask
            blend_mask = create_blend_mask(output_width, output_height, overlap_width)

            # Stitching
            stitched_equirectangular = stitch_equirectangular_pair(equi_left, equi_right, output_width, output_height, control_points_left, control_points_right, blend_mask)

            if stitched_equirectangular is not None:
                stitched_equirectangular_rgb = cv2.cvtColor(stitched_equirectangular, cv2.COLOR_BGR2RGB)
                resized_equirectangular = resize_image(stitched_equirectangular_rgb)
                st.subheader("Stitched Image")
                st.image(resized_equirectangular, caption="Stitched Equirectangular Image", use_column_width=True)
            else:
                st.error("Failed to stitch the equirectangular images.")

            # Save the stitched equirectangular image
            if st.button("Save Equirectangular Image"):
                cv2.imwrite("stitched_equirectangular.png", stitched_equirectangular)
                st.success("Equirectangular image saved as 'stitched_equirectangular.png'.")

if __name__ == "__main__":
    main()