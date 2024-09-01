import cv2
import os
from image_capture import ImageCapture
from face_recognition_model import FaceRecognitionModel
from customer import Customer

def capture_and_select_frames(num_images=10):
    """Captures and allows the user to select frames from the webcam."""
    image_capture = ImageCapture()
    images = image_capture.capture_images(num_images=num_images)
    
    saved_image_paths = []
    
    for idx, img in enumerate(images):
        filename = f"frame_{idx}.jpg"
        cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        saved_image_paths.append(filename)
        cv2.imshow(f"Captured Image {idx}", img)
    
    print("Press 'q' to finish capturing frames.")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    
    return saved_image_paths

def register_customer(customer_id, locker_number, image_paths):
    """Registers a new customer with average face embedding."""
    model = FaceRecognitionModel()
    
    # Get face embeddings from images
    images = [cv2.imread(path) for path in image_paths]
    embeddings = model.get_embeddings(images)
    
    if len(embeddings) < 5:
        print("Please capture at least 5 frames.")
        return
    
    # Compute average embedding
    avg_embedding = model.average_embedding(embeddings)
    
    # Register customer with average embedding
    customer = Customer(customer_id, locker_number)
    customer.add_customer(avg_embedding)
    
    print(f"Customer {customer_id} registered successfully with locker {locker_number}")

if __name__ == "__main__":
    customer_id = input("Enter customer ID: ")
    locker_number = input("Enter locker number: ")
    
    print("Capturing frames...")
    image_paths = capture_and_select_frames(num_images=10)
    
    if len(image_paths) >= 5:
        print("Computing face embeddings and registering customer...")
        register_customer(customer_id, locker_number, image_paths)
    else:
        print("Not enough frames captured. Please try again.")
