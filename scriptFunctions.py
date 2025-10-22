import os
import cv2
from datetime import datetime
from google.cloud import vision
from google.auth.credentials import AnonymousCredentials
from google.auth.transport.requests import Request
from google.auth.credentials import Credentials 
from datetime import datetime

"""
Image processing section
"""
def perform_ocr(image_path, api_key):
    """
    Performs OCR using Google Cloud Vision API on the given image.

    Args:
        image_path (str): The path to the image file.
        api_key (str): The Google Cloud API key.

    Returns:
        str: The extracted text from the image.
    """
    # Create a custom session with the API key
    credentials = AnonymousCredentials()
    session = Request()
    credentials = Credentials(api_key=api_key, refresh_token=None, token=None, token_uri=None)
    client_options = {
        "credentials": credentials,
        "quota_project_id": None,
    }
    
    # Instantiate a client with the API key
    client = vision.ImageAnnotatorClient(client_options=client_options)

    # Load the image file
    with open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    # Perform text detection
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if response.error.message:
        raise Exception(f'{response.error.message}\nFor more info on error messages, check: https://cloud.google.com/apis/design/errors')

    # Extract the first text annotation (the entire text block)
    if texts:
        return texts[0].description.strip()
    else:
        return "No Text"


def capture_image_from_camera(camera_index=0, frame_width=640, frame_height=480):
    """
    Capture an image from a USB camera.

    Parameters:
        camera_index (int): Index of the camera to be used for capturing (default is 0).
        frame_width (int): Width of the frame to be captured (default is 640).
        frame_height (int): Height of the frame to be captured (default is 480).

    Returns:
        numpy.ndarray: The captured image if successful, None otherwise.
    """
    # Initialize the camera
    cam = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    
    # Set the camera properties
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    
    # Additional settings can be uncommented and adjusted as needed
    # cam.set(cv2.CAP_PROP_BRIGHTNESS, <value>)
    cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
    
    # Capture the image
    success, frame = cam.read()
    cam.release()

    if success:
        print('Image capture successful.')
        return frame
    else:
        print('Image capture failed.')
        return None

def save_image(image, save_directory, image_index):
    """
    Save an image with a timestamp and index as part of the filename.

    Parameters:
        image (numpy.ndarray): The image to be saved.
        save_directory (str): The directory where the image should be saved.
        image_index (int): A unique index to append to the image filename.

    Returns:
        str: The full path to the saved image file.
    """
    # Ensure the save directory exists
    os.makedirs(save_directory, exist_ok=True)
    
    # Generate a timestamp for the filename
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    
    # Construct the full file path
    image_file_name = f"{timestamp}_{image_index}.jpg"
    image_file_path = os.path.join(save_directory, image_file_name)
    
    # Save the image
    cv2.imwrite(image_file_path, image)
    
    # Save the scan image as well
    scan_file_path = os.path.join(save_directory, 'scan.jpg')
    cv2.imwrite(scan_file_path, image)

    # Log the saved image path
    print(f'Image saved as: {image_file_path}')
    
    return scan_file_path

"""
Srting processing section
"""
def check_for_six_digit_number(lst):
    """
    Checks a list of strings for a six-digit number. If found, returns the number.
    If not found, concatenates all alphanumeric characters from the list and returns that instead.

    Parameters:
        lst (list of str): List of strings to be checked.

    Returns:
        tuple: A tuple containing either a six-digit number and None, or None and the cleaned string.
    """
    # Iterate through the list to find a six-digit number
    for element in lst:
        if element.isdigit() and len(element) == 6:
            print(f'Order number found: {element}')
            return element, None

    # Concatenate alphanumeric characters from the list if no six-digit number is found
    cleaned_str = ''.join(filter(str.isalnum, ''.join(lst)))
    print(f'No order number found, using keyword instead: {cleaned_str}')
    
    return None, cleaned_str

def build_dictionary(part_count):
    """
    Builds a dictionary with a specified number of entries, each initialized with a set of default values.

    Parameters:
        part_count (int): The number of entries (keys) in the dictionary.

    Returns:
        dict: A dictionary where each key is an index (0 to part_count-1) and the value is a dictionary 
              with predefined keys and default values.
    """
    # Create the dictionary with default values for each part
    return {
        index: {
            'order_id': '',
            'location_placed': False,
            'source': None,
            'state': None,
            'pair_found': False,
            'keyword': ''
        } for index in range(part_count)
    }

def diction_check_fill(element, cleaned_str, diction):
    """
    Checks and updates a dictionary with an order number or a keyword. If the element is found in the dictionary,
    it marks the pair as found and updates the source. If not found, it places the element or keyword in the 
    first available location.

    Parameters:
        element (str): The order number to be checked and filled in the dictionary.
        cleaned_str (str): The keyword to be checked if the order number is not found.
        diction (dict): The dictionary to be checked and updated.

    Returns:
        int: The index in the dictionary where the element or keyword was found or placed.
    """
    if element:
        for index, item in diction.items():
            if item['order_id'] == element:
                item['pair_found'] = True
                item['source'] = 'Internal'
                print(f'Order number {element} found, pair found at index {index}.')
                return index
        for index, item in diction.items():
            if not item['location_placed']:
                item['location_placed'] = True
                item['order_id'] = element
                item['source'] = 'Internal'
                print(f'Order number {element} not found, placed at new location index {index}.')
                return index
    else:
        for index, item in diction.items():
            if item['keyword'] == cleaned_str:
                item['pair_found'] = True
                item['source'] = 'External'
                print(f'Keyword {cleaned_str} found, pair found at index {index}.')
                return index
        for index, item in diction.items():
            if not item['location_placed']:
                item['location_placed'] = True
                item['keyword'] = cleaned_str
                item['source'] = 'External'
                print(f'Keyword {cleaned_str} not found, placed at new location index {index}.')
                return index

import base64
import json
import requests

def perform_ocr_with_google_cloud(image_path, api_key):
    """
    Performs OCR on the given image using the Google Cloud Vision API.
    
    Parameters:
      api_key (str): Your Google Cloud API key.
      image_path (str): Path to the image file to perform OCR on.
    
    Returns:
      dict: The JSON response from the API containing the OCR results.
    """
    # Read the image file in binary mode and encode it to base64
    with open(image_path, "rb") as image_file:
        content = base64.b64encode(image_file.read()).decode("UTF-8")
    
    # Construct the payload for the API request
    payload = {
        "requests": [
            {
                "image": {"content": content},
                "features": [{"type": "TEXT_DETECTION"}]
            }
        ]
    }
    
    # Set up the Vision API URL with your API key
    url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
    
    # Send a POST request to the Vision API
    response = requests.post(url, json=payload)
    
    # Check if the request was successful
    if response.status_code != 200:
        raise Exception(f"Request failed with status code {response.status_code}: {response.text}")
    
    # Return the JSON response containing OCR results
    return response.json()

import re

def extract_text_tokens(ocr_response):
    """
    Extracts the full recognized text from the OCR response and splits it into tokens
    using both spaces and newline characters.

    Parameters:
      ocr_response (dict): The JSON response from the Google Cloud Vision OCR API.

    Returns:
      list: A list of text tokens extracted from the OCR result.
    """
    try:
        # Get the first response item
        response_item = ocr_response['responses'][0]
        # Get the full text from the first textAnnotation
        full_text = response_item['textAnnotations'][0]['description']
        # Split the text by any whitespace (spaces, newlines, tabs, etc.)
        tokens = re.split(r'\s+', full_text.strip())
        return tokens
    except (KeyError, IndexError) as e:
        print("Error extracting text:", e)
        return []