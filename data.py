import cv2
import numpy as np

def get_object_size(image_path, object_height_cm):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect objects in the image using a pre-trained Haar Cascade classifier
    body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
    bodies = body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(bodies) == 0:
        print("No body detected.")
        return

    # Assuming only one body is detected, get its size in pixels
    x, y, w, h = bodies[0]
    object_size_pixels = w

    # Calculate the size in centimeters based on a known height of the object
    object_size_cm = (object_size_pixels / image.shape[0]) * object_height_cm

    return object_size_cm

if __name__ == "__main__":
    image_path = "your_image.jpg"  # Replace with the path to your image
    object_height_cm = 175.0  # Replace with the actual height of the object in centimeters

    object_size = get_object_size(image_path, object_height_cm)

    if object_size is not None:
        print(f"The size of the object (body) is approximately {object_size:.2f} cm.")
