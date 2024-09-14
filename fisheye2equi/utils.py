# fisheye2equi/utils.py

import logging
from PIL import Image, ExifTags
import numpy as np
import io

logger = logging.getLogger(__name__)

def extract_image_info(uploaded_file):
    image_info = {}
    try:
        # Read the image using PIL
        image = Image.open(uploaded_file)
        
        # Get basic image info
        image_info['width'], image_info['height'] = image.size
        image_info['aspect_ratio'] = image.size[0] / image.size[1]
        image_info['file_size'] = uploaded_file.size / (1024 * 1024)  # Size in MB
        
        # Extract EXIF data
        exif_data = image._getexif()
        if exif_data:
            exif = {
                ExifTags.TAGS.get(k, k): v
                for k, v in exif_data.items()
            }
            image_info['exif_data'] = exif
            image_info['camera_make'] = exif.get('Make', 'Unknown')
            image_info['camera_model'] = exif.get('Model', 'Unknown')
            image_info['datetime'] = exif.get('DateTime', 'Unknown')
            image_info['focal_length'] = exif.get('FocalLength', 'Unknown')
            image_info['f_number'] = exif.get('FNumber', 'Unknown')
            image_info['iso'] = exif.get('ISOSpeedRatings', 'Unknown')
        else:
            image_info['exif_data'] = {}
            image_info['camera_make'] = 'Unknown'
            image_info['camera_model'] = 'Unknown'
            image_info['datetime'] = 'Unknown'
            image_info['focal_length'] = 'Unknown'
            image_info['f_number'] = 'Unknown'
            image_info['iso'] = 'Unknown'
        
        return image_info
    except Exception as e:
        logger.error(f"Failed to extract image info: {str(e)}")
        return {}
