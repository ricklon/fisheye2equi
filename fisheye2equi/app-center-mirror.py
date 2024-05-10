import numpy as np
import cv2
import streamlit as st

def equidistant_to_equirectangular(fisheye_img, width, height, fov=195):
    equirectangular = np.zeros((height, width, 3), dtype=np.uint8)
    if fisheye_img is None:
        return equirectangular

    h, w = fisheye_img.shape[:2]
    cx, cy = w // 2, h // 2  # Center of the fisheye image
    r_max = min(cx, cy)  # Maximum radius of the fisheye

    for x_eq in range(width):
        theta = (x_eq / width - 0.5) * 2 * np.pi
        for y_eq in range(height):
            phi = (0.5 - y_eq / height) * np.pi  # Corrected phi calculation
            r = r_max * phi / (fov * np.pi / 360)
            x_fish = int(cx + r * np.cos(theta))
            y_fish = int(cy + r * np.sin(theta))
            if 0 <= x_fish < w and 0 <= y_fish < h:
                equirectangular[y_eq, x_eq, :] = fisheye_img[y_fish, x_fish, :]

    return equirectangular

def visualize_overlap(fisheye_img, fov=195, overlap_range=(180, 195)):
    h, w = fisheye_img.shape[:2]
    cx, cy = w // 2, h // 2  # Center of the fisheye image
    r_max = min(cx, cy)  # Maximum radius of the fisheye

    overlap_img = fisheye_img.copy()

    for angle in range(overlap_range[0], overlap_range[1] + 1, 5):
        r = r_max * angle / fov
        cv2.circle(overlap_img, (cx, cy), int(r), (0, 255, 0), 2)

    return overlap_img

def stitch_equirectangular_pair(equi1, equi2, width, height):
    if equi1 is None or equi2 is None:
        return None

    equirectangular = np.zeros((height, width, 3), dtype=np.uint8)
    equirectangular[:, :width//2, :] = equi1
    equirectangular[:, width//2:, :] = equi2

    return equirectangular

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

        # Visualize radial overlap
        overlap_left = visualize_overlap(fisheye_left, fov)
        overlap_right = visualize_overlap(fisheye_right, fov)

        # Resize the overlap images for display
        resized_overlap_left = resize_image(cv2.cvtColor(overlap_left, cv2.COLOR_BGR2RGB))
        resized_overlap_right = resize_image(cv2.cvtColor(overlap_right, cv2.COLOR_BGR2RGB))

        st.image([resized_overlap_left, resized_overlap_right], caption=["Radial Overlap (Left)", "Radial Overlap (Right)"], width=resized_overlap_left.shape[1])

        # User confirmation for stitching
        stitch_confirmed = st.checkbox("Confirm Overlap and Proceed with Stitching")

        if stitch_confirmed:
            # Equidistant to equirectangular projection
            equi_left = equidistant_to_equirectangular(fisheye_left, output_width // 2, output_height, fov)
            equi_right = equidistant_to_equirectangular(fisheye_right, output_width // 2, output_height, fov)

            # Resize the equirectangular projections for display
            resized_equi_left = resize_image(cv2.cvtColor(equi_left, cv2.COLOR_BGR2RGB))
            resized_equi_right = resize_image(cv2.cvtColor(equi_right, cv2.COLOR_BGR2RGB))

            st.image([resized_equi_left, resized_equi_right], caption=["Equirectangular Projection (Left)", "Equirectangular Projection (Right)"], width=resized_equi_left.shape[1])

            # Stitching
            stitched_equirectangular = stitch_equirectangular_pair(equi_left, equi_right, output_width, output_height)
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