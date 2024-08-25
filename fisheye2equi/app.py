import streamlit as st
from PIL import Image
import numpy as np
import cv2
import sys
import os
import io

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fisheye2equi.utils import extract_image_info
from fisheye2equi.stitching import stitch_gear360_image
from logging_config import setup_logging  # Import the logging setup

def main():
    st.title("Gear 360 Image Stitcher")

    # Set up logging
    debug_mode = st.sidebar.checkbox("Enable Debug Mode")
    logger = setup_logging(debug_mode)
    logger.info("Gear 360 Image Stitcher started")

    # Create two columns for method selection
    col1, col2 = st.columns(2)
    
    with col1:
        rotation_method = st.selectbox("Select Rotation Method", ['simple', 'advanced'])
    
    with col2:
        stitching_method = st.selectbox("Select Stitching Method", ['simple', 'advanced'])

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
                    st.write(f"Camera Make: {image_info['camera_make']}")
                    st.write(f"Camera Model: {image_info['camera_model']}")
                    st.write(f"Date/Time: {image_info['datetime']}")
                    st.write(f"Focal Length: {image_info['focal_length']}")
                    st.write(f"F-Number: {image_info['f_number']}")
                    st.write(f"ISO: {image_info['iso']}")
            
            # Determine which .pto profile to use based on image info
            if image_info['camera_model'] == 'SM-C200':
                pto_profile = 'gear360sm-c200.pto'
            elif image_info['camera_model'] == 'SM-R210':
                pto_profile = 'gear360sm-r210.pto'
            else:
                logger.warning(f"Unsupported camera model: {image_info['camera_model']}. Using default profile.")
                st.warning(f"Unsupported camera model: {image_info['camera_model']}. Using default profile.")
                pto_profile = 'gear360sm-c200.pto'  # Use a default profile
            
            # Stitch button
            if st.button("Stitch Image"):
                logger.info(f"Stitching process started with rotation method: {rotation_method}, stitching method: {stitching_method}")
                st.write("Stitching in progress...")
                try:
                    stitched_image, debug_images = stitch_gear360_image(opencv_image, pto_profile, image_info, 
                                                                        rotation_method=rotation_method, 
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
                        if debug_mode and debug_images:
                            logger.debug("Displaying debug images")
                            st.subheader("Debug Images")
                            
                            # Create a 2x2 grid for debug images
                            col1, col2 = st.columns(2)
                            col3, col4 = st.columns(2)
                            
                            for i, (title, image) in enumerate(debug_images.items()):
                                if i % 4 == 0:
                                    col1.image(image, caption=title, use_column_width=True)
                                elif i % 4 == 1:
                                    col2.image(image, caption=title, use_column_width=True)
                                elif i % 4 == 2:
                                    col3.image(image, caption=title, use_column_width=True)
                                else:
                                    col4.image(image, caption=title, use_column_width=True)
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