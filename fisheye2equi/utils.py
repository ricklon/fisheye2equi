from PIL import Image
import exif

def extract_image_info(image_file):
    """
    Extract metadata and size information from an uploaded image file.
    
    :param image_file: UploadedFile object from Streamlit
    :return: Dictionary containing image information
    """
    info = {}
    
    # Get file size
    info['file_size'] = image_file.size / (1024 * 1024)  # Size in MB
    
    # Open image and get dimensions
    img = Image.open(image_file)
    info['width'], info['height'] = img.size
    info['aspect_ratio'] = info['width'] / info['height']
    
    # Extract EXIF metadata
    image_file.seek(0)  # Reset file pointer to beginning
    exif_image = exif.Image(image_file)
    if exif_image.has_exif:
        info['camera_make'] = exif_image.get('make', 'Unknown')
        info['camera_model'] = exif_image.get('model', 'Unknown')
        info['datetime'] = exif_image.get('datetime', 'Unknown')
        info['focal_length'] = exif_image.get('focal_length', 'Unknown')
        info['f_number'] = exif_image.get('f_number', 'Unknown')
        info['iso'] = exif_image.get('iso', 'Unknown')
    else:
        info['exif_data'] = 'No EXIF data found'
    
    return info
 