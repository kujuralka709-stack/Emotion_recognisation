# 😊 AI-Based Emotion Recognition Using Deep Learning

## 📌 Project Overview

The AI-Based Emotion Recognition System is a Python project that uses Artificial Intelligence (AI), Deep Learning, and Computer Vision to detect faces and classify facial expressions in real time using a webcam.

The project uses OpenCV for face detection and a pre-trained TensorFlow/Keras model for facial-expression classification.

## 🎯 Objectives

- Detect faces through a webcam.
- Recognize facial-expression categories in real time.
- Understand the practical application of AI and deep learning.
- Build a beginner-friendly computer vision project.

## ✨ Features

- Real-time webcam access.
- Automatic face detection.
- Facial-expression classification.
- Displays the predicted category and model confidence.
- Simple Python implementation.

## 🛠️ Technologies Used

- **Python** — Programming language
- **OpenCV** — Face detection and webcam processing
- **TensorFlow/Keras** — Deep learning model
- **NumPy** — Image data processing
- **Haar Cascade** — Face detection method

## 📂 Project Structure

```text
Emotion_recognisation/
│
├── emotion_recognisation.py
├── face_detection.py
├── finalfacialemotionmodel.keras
└── README.md
```

## ⚙️ Installation

### Step 1: Install Python

Install a compatible Python version and verify the installation:

```bash
python --version
```

### Step 2: Install Required Libraries

Open the terminal in VS Code and run:

```bash
python -m pip install opencv-python numpy tensorflow
```

### Step 3: Download the Model

Download the trained model file `finalfacialemotionmodel.keras` and place it in the project folder. The program requires this file to make predictions.

### Step 4: Run the Project

Start the emotion recognition program:

```bash
python emotion_recognisation.py
```

Allow webcam access if prompted. Press **Q** to close the webcam window.

## 🔍 How It Works

1. The webcam captures a video frame.
2. OpenCV detects the face.
3. The face image is converted to grayscale and resized to 48 × 48 pixels.
4. The trained deep learning model processes the image.
5. The program displays the predicted facial-expression category.

## 📊 Example Output

The webcam window displays a detected face with a predicted category, such as:

- Happy
- Sad
- Angry
- Fear
- Surprise
- Disgust
- Neutral

The exact categories depend on the model's training labels.

## 🔮 Future Scope

- Improve model accuracy through testing.
- Add a user-friendly graphical interface.
- Optimize performance for low-powered devices.
- Explore privacy-friendly, on-device processing.
- Evaluate performance under different lighting conditions.

## 🚀 Applications

- Computer vision learning and research.
- Human-computer interaction experiments.
- Facial-expression analysis prototypes.
- AI and deep learning education.

## ⚠️ Limitations

- Predictions may be incorrect in some conditions.
- Lighting, camera quality, and face position can affect performance.
- Facial expressions do not reliably reveal a person's true emotional state.
- The model's accuracy and output labels should be verified before practical use.

## 🔐 Privacy

Use the webcam only with appropriate permission. Avoid recording or sharing identifiable face images without consent.

## 👩‍💻 Author

**Alka Kujur**

## 📜 License

This project is intended for educational purposes. Check the original model's license and usage terms before redistributing the model or using it commercially.
