import numpy as np
import cv2
import streamlit as st

def equisolid_to_equirectangular(fisheye_img, width, height, overlap_angle):
    equirectangular = np.zeros((height, width, 3), dtype=np.uint8)
    if fisheye_img is None:
        return equirectangular

    h, w = fisheye_img.shape[:2]
    cx, cy = w // 2, h // 2  # Center of the fisheye image
    r_max = min(cx, cy)  # Maximum radius of the fisheye

    overlap_pixels = int(width * overlap_angle / 360)

    for x_eq in range(width):
        theta = 2 * np.pi * (x_eq - overlap_pixels) / (width - 2 * overlap_pixels)  # 0 to 2pi
        for y_eq in range(height):
            phi = np.pi * y_eq / height - np.pi / 2  # -pi/2 to pi/2
            r = 2 * r_max * np.sin(phi / 2)  # Equisolid angle projection
            fx = int(cx + r * np.cos(theta))
            fy = int(cy + r * np.sin(theta))
            if 0 <= fx < w and 0 <= fy < h:
                equirectangular[y_eq, x_eq, :] = fisheye_img[fy, fx, :]

    return equirectangular

def stitch_fisheye_pair(fisheye1, fisheye2, width, height, overlap_angle):
    if fisheye1 is None or fisheye2 is None:
        return None

    # Convert fisheye images from BGR to RGB
    fisheye1 = cv2.cvtColor(fisheye1, cv2.COLOR_BGR2RGB)
    fisheye2 = cv2.cvtColor(fisheye2, cv2.COLOR_BGR2RGB)

    # Apply equirectangular projection to each fisheye image
    equi1 = equisolid_to_equirectangular(fisheye1, width // 2, height, overlap_angle)
    equi2 = equisolid_to_equirectangular(fisheye2, width // 2, height, -overlap_angle)

    # Determine the stitching seam based on the overlap angle
    overlap_pixels = int(width * abs(overlap_angle) / 360)
    seam_position = width // 2

    # Create the final equirectangular image
    equirectangular = np.zeros((height, width, 3), dtype=np.uint8)

    # Stitch the equirectangular projections
    equirectangular[:, :seam_position - overlap_pixels] = equi1[:, :seam_position - overlap_pixels]
    equirectangular[:, seam_position + overlap_pixels:] = equi2[:, seam_position - overlap_pixels:]

    # Apply blending to minimize the seam
    blend_width = overlap_pixels // 2
    if blend_width > 0:
        left_blend = equirectangular[:, seam_position - overlap_pixels - blend_width:seam_position - overlap_pixels]
        right_blend = equirectangular[:, seam_position + overlap_pixels:seam_position + overlap_pixels + blend_width]
        alpha = np.linspace(0, 1, blend_width)
        alpha_mask = np.repeat(alpha[np.newaxis, :], height, axis=0)
        alpha_mask = np.repeat(alpha_mask[:, :, np.newaxis], 3, axis=2)
        equirectangular[:, seam_position - overlap_pixels - blend_width:seam_position - overlap_pixels] = (
            left_blend * (1 - alpha_mask) + right_blend * alpha_mask
        ).astype(np.uint8)

    return equirectangular

def get_image_properties(image):
    height, width = image.shape[:2]
    aspect_ratio = width / height
    properties = {
        "Width": width,
        "Height": height,
        "Aspect Ratio": aspect_ratio
    }
    return properties

def suggest_output_sizes(image):
    properties = get_image_properties(image)
    width = properties["Width"]
    height = properties["Height"]

    suggested_sizes = []

    if width == 3840 and height == 1920:  # Gear 360 2017
        suggested_sizes = [
            (4096, 2048),  # 4K
            (2048, 1024),  # 2K
            (1920, 1080)   # 1080p
        ]
    elif width == 5120 and height == 2560:  # Gear 360 2016
        suggested_sizes = [
            (7776, 3888),  # 30 megapixels
            (5472, 2736),  # 15 megapixels
            (3840, 1920)   # 8 megapixels
        ]
    else:
        suggested_sizes = [
            (width, height // 2),
            (width // 2, height // 4),
            (width // 4, height // 8)
        ]

    return suggested_sizes

def draw_overlap_lines(fisheye_img, overlap_angle, color=(0, 255, 0), thickness=2):
    h, w = fisheye_img.shape[:2]
    cx, cy = w // 2, h // 2
    r = min(cx, cy)

    # Calculate the angle range based on the overlap angle
    start_angle = -overlap_angle / 2
    end_angle = overlap_angle / 2

    # Convert angles from degrees to radians
    start_angle_rad = np.deg2rad(start_angle)
    end_angle_rad = np.deg2rad(end_angle)

    # Calculate the endpoints of the overlap lines
    start_x = int(cx + r * np.cos(start_angle_rad))
    start_y = int(cy + r * np.sin(start_angle_rad))
    end_x = int(cx + r * np.cos(end_angle_rad))
    end_y = int(cy + r * np.sin(end_angle_rad))

    # Draw the overlap lines on the fisheye image
    cv2.line(fisheye_img, (cx, cy), (start_x, start_y), color, thickness)
    cv2.line(fisheye_img, (cx, cy), (end_x, end_y), color, thickness)

    return fisheye_img

def main():
    st.title("Fisheye Image Stitching")

    # File upload
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the uploaded image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        combined_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Convert the uploaded image from BGR to RGB
        combined_image_rgb = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)

        # Display the uploaded image
        st.image(combined_image_rgb, caption="Uploaded Image", use_column_width=True)

        # Get image properties
        properties = get_image_properties(combined_image)
        st.write("Image Properties:")
        st.write(properties)

        # Suggest output sizes
        suggested_sizes = suggest_output_sizes(combined_image)
        st.write("Suggested Output Sizes:")
        selected_size = st.selectbox("Select an output size", options=suggested_sizes)

        # Input fields for modification
        overlap_angle = st.slider("Overlap Angle", min_value=-180, max_value=180, value=0, step=1)
        output_width = selected_size[0]
        output_height = selected_size[1]

        if st.button("Preview Equirectangular Projections"):
            # Show progress bar
            progress_text = "Processing..."
            progress_bar = st.progress(0, text=progress_text)

            # Split the combined image into two fisheye images
            h, w = combined_image.shape[:2]
            fisheye1 = combined_image[:, :w // 2, :]
            fisheye2 = combined_image[:, w // 2:, :]

            # Draw overlap lines on the fisheye images
            fisheye1_overlap = draw_overlap_lines(fisheye1, overlap_angle)
            fisheye2_overlap = draw_overlap_lines(fisheye2, -overlap_angle)

            # Apply equirectangular projection to each fisheye image
            with st.spinner("Applying equirectangular projection..."):
                equi1 = equisolid_to_equirectangular(fisheye1_overlap, output_width // 2, output_height, overlap_angle)
                equi2 = equisolid_to_equirectangular(fisheye2_overlap, output_width // 2, output_height, -overlap_angle)

            # Update progress bar
            progress_bar.progress(1, text="Processing complete!")

            # Convert the fisheye images from BGR to RGB
            fisheye1_overlap = cv2.cvtColor(fisheye1_overlap, cv2.COLOR_BGR2RGB)
            fisheye2_overlap = cv2.cvtColor(fisheye2_overlap, cv2.COLOR_BGR2RGB)

            # Resize the fisheye images to match the size of the original image
            fisheye1_resized = cv2.resize(fisheye1_overlap, (w // 2, h))
            fisheye2_resized = cv2.resize(fisheye2_overlap, (w // 2, h))

            # Display the fisheye images with overlap lines side by side
            st.image([fisheye1_resized, fisheye2_resized], caption=["Fisheye 1 with Overlap", "Fisheye 2 with Overlap"], width=w // 2)

            # Convert the equirectangular projections from BGR to RGB
            equi1 = cv2.cvtColor(equi1, cv2.COLOR_BGR2RGB)
            equi2 = cv2.cvtColor(equi2, cv2.COLOR_BGR2RGB)

            # Resize the equirectangular projections to match the size of the original image
            equi1_resized = cv2.resize(equi1, (w // 2, h // 2))
            equi2_resized = cv2.resize(equi2, (w // 2, h // 2))

            # Display the equirectangular projections side by side
            st.image([equi1_resized, equi2_resized], caption=["Equirectangular Projection 1", "Equirectangular Projection 2"], width=w // 2)
        
        if st.button("Convert"):
            # Show progress bar
            progress_text = "Processing..."
            progress_bar = st.progress(0, text=progress_text)

            # Split the combined image into two fisheye images
            h, w = combined_image.shape[:2]
            fisheye1 = combined_image[:, :w // 2, :]
            fisheye2 = combined_image[:, w // 2:, :]

            # Stitch the fisheye images
            with st.spinner("Stitching images..."):
                try:
                    equirectangular_img = stitch_fisheye_pair(fisheye1, fisheye2, output_width, output_height, overlap_angle)
                except ValueError as e:
                    st.error(str(e))
                    progress_bar.progress(1, text="Processing failed.")
                    return
            
            # Update progress bar
            progress_bar.progress(1, text="Processing complete!")

            if equirectangular_img is not None:
                # Display the stitched image
                st.image(equirectangular_img, caption="Stitched Equirectangular Image", use_column_width=True)

                # Download button for the stitched image
                _, buffer = cv2.imencode(".jpg", equirectangular_img)
                st.download_button(
                    label="Download Stitched Image",
                    data=buffer.tobytes(),
                    file_name="stitched_image.jpg",
                    mime="image/jpeg"
                )
            else:
                st.error("Failed to stitch the fisheye images.")

if __name__ == "__main__":
    main()