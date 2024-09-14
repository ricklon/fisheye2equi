import streamlit as st
from PIL import Image
import numpy as np
import cv2
import sys
import os
import io

# Adjust the import paths as necessary
from fisheye2equi.stitching import stitch_gear360_image
from fisheye2equi.utils import extract_image_info
from logging_config import setup_logging

def main():
    st.title("Gear 360 Image Stitcher")

    # Set up logging with debug mode on by default
    debug_mode = st.sidebar.checkbox("Debug Mode", value=True)
    logger = setup_logging(debug_mode)
    logger.info("Gear 360 Image Stitcher started")

    # Since rotation method is now handled inside fisheye_to_equirectangular, we can remove the rotation method selection
    stitching_method = st.sidebar.radio("Select Stitching Method", ['simple'])  # Currently, only 'simple' is implemented

    uploaded_file = st.file_uploader("Choose a Gear 360 image file", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            # Check file size
            file_size = uploaded_file.size / (1024 * 1024)  # Size in MB
            if file_size > 200:
                logger.error(f"File size ({file_size:.2f} MB) exceeds the 200 MB limit.")
                st.error(f"File size ({file_size:.2f} MB) exceeds the 200 MB limit.")
                return

            # Attempt to open the image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if opencv_image is None:
                logger.error("Failed to read the image. The file may be corrupted or in an unsupported format.")
                st.error("Failed to read the image. The file may be corrupted or in an unsupported format.")
                return
            
            # Display the uploaded image
            st.image(cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB), caption="Uploaded Gear 360 Image", use_column_width=True)
            
            # Extract and display image info
            image_info = extract_image_info(uploaded_file)
            logger.info(f"Image info extracted: {image_info}")
            
            # Create two columns for image info
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Image Information")
                st.write(f"Dimensions: {image_info['width']}x{image_info['height']} pixels")
                st.write(f"Aspect Ratio: {image_info['aspect_ratio']:.2f}")
                st.write(f"File Size: {image_info['file_size']:.2f} MB")
            
            with col2:
                st.subheader("EXIF Metadata")
                if 'exif_data' in image_info:
                    st.write(image_info['exif_data'])
                else:
                    st.write(f"Camera Make: {image_info.get('camera_make', 'Unknown')}")
                    st.write(f"Camera Model: {image_info.get('camera_model', 'Unknown')}")
                    st.write(f"Date/Time: {image_info.get('datetime', 'Unknown')}")
                    st.write(f"Focal Length: {image_info.get('focal_length', 'Unknown')}")
                    st.write(f"F-Number: {image_info.get('f_number', 'Unknown')}")
                    st.write(f"ISO: {image_info.get('iso', 'Unknown')}")
            
            # Determine which .pto profile to use based on image info
            if image_info.get('camera_model') == 'SM-C200':
                pto_profile = 'gear360sm-c200.pto'
            elif image_info.get('camera_model') == 'SM-R210':
                pto_profile = 'gear360sm-r210.pto'
            else:
                logger.warning(f"Unsupported camera model: {image_info.get('camera_model', 'Unknown')}. Using default profile.")
                st.warning(f"Unsupported camera model: {image_info.get('camera_model', 'Unknown')}. Using default profile.")
                pto_profile = 'gear360sm-c200.pto'  # Use a default profile
            
            # Stitch button
            if st.button("Stitch Image"):
                logger.info(f"Stitching process started with stitching method: {stitching_method}")
                st.write("Stitching in progress...")
                try:
                    stitched_image, debug_images = stitch_gear360_image(opencv_image, pto_profile, image_info,
                                                                        stitching_method=stitching_method,
                                                                        debug=debug_mode)
                    
                    if stitched_image is not None:
                        st.image(cv2.cvtColor(stitched_image, cv2.COLOR_BGR2RGB), caption="Stitched Equirectangular Image", use_column_width=True)
                        logger.info("Stitching completed successfully")
                        
                        # Option to download the stitched image
                        buffered = io.BytesIO()
                        Image.fromarray(cv2.cvtColor(stitched_image, cv2.COLOR_BGR2RGB)).save(buffered, format="JPEG")
                        st.download_button(
                            label="Download Stitched Image",
                            data=buffered.getvalue(),
                            file_name="stitched_gear360_image.jpg",
                            mime="image/jpeg"
                        )
                        
                        # Display debug images if debug mode is enabled
                        if debug_images:
                            logger.debug("Displaying debug images")
                            st.subheader("Debug Images")
                            
                            # Create a 3x3 grid for debug images
                            cols = st.columns(3)
                            for i, (title, image) in enumerate(debug_images.items()):
                                cols[i % 3].image(image, caption=title, use_column_width=True)
                    else:
                        logger.error("Stitching failed. The image may not be compatible or there might be an issue with the stitching process.")
                        st.error("Stitching failed. The image may not be compatible or there might be an issue with the stitching process.")
                except Exception as e:
                    logger.error(f"An error occurred during stitching: {str(e)}", exc_info=True)
                    st.error(f"An error occurred during stitching: {str(e)}")
        
        except Exception as e:
            logger.error(f"An error occurred while processing the file: {str(e)}", exc_info=True)
            st.error(f"An error occurred while processing the file: {str(e)}")

if __name__ == "__main__":
    main()
