
import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "finalfacialemotionmodel.keras"

# Load the trained emotion model
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Standard FER-2013 emotion labels
EMOTIONS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]

# Load OpenCV face detector
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_detector.empty():
    raise RuntimeError("Could not load the face detector.")

# Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Webcam not available. Check camera connection.")

try:
    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read webcam frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(40, 40)
        )

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (48, 48))
            face = face.astype("float32") / 255.0
            face = np.expand_dims(face, axis=(0, -1))

            prediction = model.predict(face, verbose=0)
            emotion_index = int(np.argmax(prediction[0]))

            # Display only if model output matches seven labels
            if prediction.shape[-1] != len(EMOTIONS):
                raise ValueError(
                    f"Model returns {prediction.shape[-1]} classes, "
                    f"but this code expects {len(EMOTIONS)}."
                )

            emotion = EMOTIONS[emotion_index]
            confidence = float(prediction[0][emotion_index]) * 100

            cv2.rectangle(
                frame, (x, y), (x + w, y + h), (0, 255, 0), 2
            )

            label = f"{emotion}: {confidence:.1f}%"
            cv2.putText(
                frame,
                label,
                (x, max(y - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        cv2.imshow("AI Emotion Recognition - Press Q to Quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
