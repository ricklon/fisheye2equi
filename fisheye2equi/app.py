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

def fisheye_to_equirectangular(fisheye_img, width, height, fov=195):
    equirectangular = np.zeros((height, width, 3), dtype=np.uint8)
    if fisheye_img is None:
        return equirectangular

    h, w = fisheye_img.shape[:2]
    cx, cy = w // 2, h // 2  # Center of the fisheye image
    r_max = min(cx, cy)  # Maximum radius of the fisheye

    for y_eq in range(height):
        theta = (y_eq / height - 0.5) * np.pi
        for x_eq in range(width):
            phi = (0.5 - x_eq / width) * 2 * np.pi

            # Calculate fisheye radius using polynomial coefficients
            r = P1 * phi**5 + P2 * phi**4 + P3 * phi**3 + P4 * phi**2 + P5 * phi + P6
            r *= r_max

            # Calculate fisheye coordinates
            x_fish = int(cx + r * np.cos(theta))
            y_fish = int(cy + r * np.sin(theta))

            if 0 <= x_fish < w and 0 <= y_fish < h:
                equirectangular[y_eq, x_eq, :] = fisheye_img[y_fish, x_fish, :]

    return equirectangular

def compensate_light(equirectangular_img, clip_limit=2.0, tile_grid_size=(8, 8)):
    if equirectangular_img is None:
        return None

    # Convert the image to the LAB color space
    lab_img = cv2.cvtColor(equirectangular_img, cv2.COLOR_BGR2LAB)

    # Split the LAB image into L, A, and B channels
    l_channel, a_channel, b_channel = cv2.split(lab_img)

    # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to the L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_channel_equalized = clahe.apply(l_channel)

    # Merge the equalized L channel back with the A and B channels
    lab_img_equalized = cv2.merge((l_channel_equalized, a_channel, b_channel))

    # Convert the image back to the BGR color space
    equirectangular_img_equalized = cv2.cvtColor(lab_img_equalized, cv2.COLOR_LAB2BGR)

    return equirectangular_img_equalized

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



def stitch_equirectangular_pair(equi_left, equi_right, width, height, control_points_left=None, control_points_right=None, blend_mask=None):
    print("Debug before stitching:")
    print("equi_left shape:", equi_left.shape)
    print("equi_right shape:", equi_right.shape)
    print("blend_mask shape:", blend_mask.shape)

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
    if combined_image.shape[1] != blend_mask.shape[1]:
        print("Resizing blend mask to match combined image width.")
        blend_mask = cv2.resize(blend_mask, (combined_image.shape[1], combined_image.shape[0]))

    stitched = combined_image * blend_mask
    stitched = np.clip(stitched, 0, 255).astype(np.uint8)

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

        # Input fields for output size and FOV
        output_width = st.number_input("Output Width", value=4096, min_value=1024, max_value=8192, step=1)
        output_height = output_width // 2
        fov = st.number_input("Field of View (FOV)", value=195, min_value=180, max_value=220, step=1)

        # Checkbox for enabling light compensation
        enable_light_compensation = st.checkbox("Enable Light Compensation")

        # Checkbox for enabling refined alignment
        enable_refined_alignment = st.checkbox("Enable Refined Alignment")

        # Slider for adjusting overlap width
        overlap_width = st.slider("Overlap Width", min_value=50, max_value=500, value=200, step=10)

        # "Go" button to initiate the stitching process
        if st.button("Go"):
            equi_left = fisheye_to_equirectangular(fisheye_left, output_width // 2, output_height, fov)
            equi_right = fisheye_to_equirectangular(fisheye_right, output_width // 2, output_height, fov)

            # Display the equirectangular images
            st.image(cv2.cvtColor(equi_left, cv2.COLOR_BGR2RGB), caption="Left Equirectangular Image", use_column_width=True)
            st.image(cv2.cvtColor(equi_right, cv2.COLOR_BGR2RGB), caption="Right Equirectangular Image", use_column_width=True)

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

            if enable_light_compensation:
                equi_left = compensate_light(equi_left)
                equi_right = compensate_light(equi_right)

            # Generate the blend mask
            blend_mask = create_blend_mask(output_width, output_height, overlap_width)

            # Stitching
            stitched_equirectangular = stitch_equirectangular_pair(equi_left, equi_right, output_width, output_height, control_points_left, control_points_right, blend_mask)

            if stitched_equirectangular is not None:
                stitched_equirectangular_rgb = cv2.cvtColor(stitched_equirectangular, cv2.COLOR_BGR2RGB)
                resized_equirectangular = resize_image(stitched_equirectangular_rgb)
                st.image(resized_equirectangular, caption="Stitched Equirectangular Image", use_column_width=True)
            else:
                st.error("Failed to stitch the equirectangular images.")

            # Save the stitched equirectangular image
            if st.button("Save Equirectangular Image"):
                cv2.imwrite("stitched_equirectangular.png", stitched_equirectangular)
                st.success("Equirectangular image saved as 'stitched_equirectangular.png'.")

if __name__ == "__main__":
    main()