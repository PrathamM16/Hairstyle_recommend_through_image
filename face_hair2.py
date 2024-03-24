import numpy as np
import cv2
import dlib
import pickle
from sklearn.ensemble import RandomForestClassifier

# Load model paths
face_cascade_path = "E:\\3RD_TRIMISTER\\FaceShape-master\\haarcascade_frontalface_default.xml"
predictor_path = "E:\\3RD_TRIMISTER\\FaceShape-master\\shape_predictor_68_face_landmarks(1).dat"
model_path = "E:\\3RD_TRIMISTER\\FaceShape-master\\face_model.pkl"

# Load face cascade, shape predictor, and trained model
faceCascade = cv2.CascadeClassifier(face_cascade_path)
predictor = dlib.shape_predictor(predictor_path)
model = None  # Initialize model as None

def load_model():
    global model
    if model is None:
        try:
            with open(model_path, "rb") as f:
                print("Loading model from:", model_path)  # Add this line
                model = pickle.load(f)
            print("Model loaded successfully!")  # Add this line
        except FileNotFoundError:
            print("Error: Face shape classification model not found. Please train and save it before using.")
            exit()
        except Exception as e:
            print("Error loading model:", e)
            exit()


def extract_face_features(landmarks):
    jawline_width = landmarks[16, 0] - landmarks[0, 0]
    forehead_length = landmarks[3, 1] - landmarks[0, 1]
    chin_roundness = abs((landmarks[8, 0] - landmarks[1, 0]) / (landmarks[15, 0] - landmarks[0, 0]))

    features = [jawline_width, forehead_length, chin_roundness]
    return features

def predict_face_shape(features):
    load_model()
    if model is not None:
        face_shape = model.predict([features])[0]
        return face_shape
    else:
        return "Model not loaded"

def detect_face_shape(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(100, 100))

    for (x, y, w, h) in faces:
        dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        detected_landmarks = predictor(gray, dlib_rect).parts()
        landmarks = np.matrix([[p.x, p.y] for p in detected_landmarks])

        face_features = extract_face_features(landmarks)
        face_shape = predict_face_shape(face_features)

        for idx, point in enumerate(landmarks):
            cv2.circle(image, (point[0, 0], point[0, 1]), 2, (0, 255, 0), -1)

        cv2.putText(image, f"Face Shape: {face_shape}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        user_choice = input("Enter your desired hairstyle (optional): ")
        overlay_hairstyle(image, face_shape, user_choice or face_shape, x, y, w, h)

    return image

def overlay_hairstyle(image, face_shape, chosen_hairstyle, x, y, w, h):
    hairstyle_images = {
        'Heart': cv2.imread("E:\3RD_TRIMISTER\FaceShape-master\Images\heart.png", cv2.IMREAD_UNCHANGED),
        'Oval': cv2.imread("E:\3RD_TRIMISTER\FaceShape-master\Images\oval.png", cv2.IMREAD_UNCHANGED),
        'Long': cv2.imread("E:\3RD_TRIMISTER\FaceShape-master\Images\long.png", cv2.IMREAD_UNCHANGED),
        'Round': cv2.imread("E:\3RD_TRIMISTER\FaceShape-master\Images\round.png", cv2.IMREAD_UNCHANGED),
        'Square': cv2.imread("E:\3RD_TRIMISTER\FaceShape-master\Images\square.png", cv2.IMREAD_UNCHANGED)
    }

    hairstyle_img = hairstyle_images.get(chosen_hairstyle)

    if hairstyle_img is not None:
        hairstyle_img_resized = cv2.resize(hairstyle_img, (w, h))

        if hairstyle_img_resized.shape[2] == 4: 
            for i in range(h):
                for j in range(w):
                    if hairstyle_img_resized[i, j, 3] != 0:  
                        image[y + i, x + j] = hairstyle_img_resized[i, j, :3]  
        else:
            image[y:y+h, x:x+w] = hairstyle_img_resized

if __name__ == "__main__":
    option = input("Choose an option (1. Live Photo Capture, 2. Image Upload): ")

    if option == '1':
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        original_image = frame
    elif option == '2':
        image_path = input("Enter the image path: ")
        original_image = cv2.imread(image_path)
    else:
        print("Invalid option. Please choose either 1 or 2.")
        exit()

    if original_image is None:
        print("Error: Could not open or read the image.")
        exit()

    image_with_shape = original_image.copy()
    image_with_hairstyle = detect_face_shape(image_with_shape)

    cv2.imshow("Original Image", original_image)
    cv2.imshow("Image with Recommended Hairstyles", image_with_hairstyle)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
