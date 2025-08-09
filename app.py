import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import io

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("emotion_detection_model.h5")
    return model

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load Haar Cascade for face detection
@st.cache_resource
def load_face_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_image(image):
    """Preprocess image for emotion prediction"""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Resize to 48x48
    image = cv2.resize(image, (48, 48))
    
    # Normalize
    image = image.astype('float32') / 255.0
    
    # Reshape for model input
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)
    
    return image

def predict_emotion(model, image):
    """Predict emotion from preprocessed image"""
    prediction = model.predict(image)
    emotion_idx = np.argmax(prediction)
    confidence = np.max(prediction)
    return emotion_labels[emotion_idx], confidence

def detect_faces_and_emotions(model, face_cascade, image):
    """Detect faces and predict emotions"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    results = []
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        preprocessed_face = preprocess_image(face_roi)
        emotion, confidence = predict_emotion(model, preprocessed_face)
        results.append({
            'bbox': (x, y, w, h),
            'emotion': emotion,
            'confidence': confidence
        })
    
    return results

def draw_emotion_results(image, results):
    """Draw bounding boxes and emotion labels on image"""
    image_copy = image.copy()
    for result in results:
        x, y, w, h = result['bbox']
        emotion = result['emotion']
        confidence = result['confidence']
        
        # Draw bounding box
        cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Draw emotion label
        label = f"{emotion}: {confidence:.2f}"
        cv2.putText(image_copy, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return image_copy

# Streamlit App
def main():
    st.set_page_config(
        page_title="Facial Emotion Detection",
        page_icon="😊",
        layout="wide"
    )
    
    st.title("🎭 Facial Emotion Detection Application")
    st.markdown("---")
    
    # Load model and face cascade
    model = load_model()
    face_cascade = load_face_cascade()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📷 Live Camera Detection", "🖼️ Upload Image(s)", "📊 Model Info & Accuracy"])
    
    with tab1:
        st.header("Live Camera Detection")
        st.markdown("Use your webcam for real-time emotion detection")
        
        # Camera input
        camera_input = st.camera_input("Take a picture")
        
        if camera_input is not None:
            # Convert to PIL Image
            image = Image.open(camera_input)
            image_array = np.array(image)
            
            # Detect faces and emotions
            results = detect_faces_and_emotions(model, face_cascade, image_array)
            
            if results:
                # Draw results
                result_image = draw_emotion_results(image_array, results)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Original Image")
                    st.image(image, use_column_width=True)
                
                with col2:
                    st.subheader("Detection Results")
                    st.image(result_image, use_column_width=True)
                
                # Display results table
                st.subheader("Detected Emotions")
                results_df = pd.DataFrame([
                    {
                        'Face': i+1,
                        'Emotion': result['emotion'],
                        'Confidence': f"{result['confidence']:.2%}"
                    }
                    for i, result in enumerate(results)
                ])
                st.dataframe(results_df, use_container_width=True)
            else:
                st.warning("No faces detected in the image. Please try again with a clearer image.")
    
    with tab2:
        st.header("Upload Image(s)")
        st.markdown("Upload single or multiple images for emotion detection")
        
        uploaded_files = st.file_uploader(
            "Choose image files",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            for i, uploaded_file in enumerate(uploaded_files):
                st.subheader(f"Image {i+1}: {uploaded_file.name}")
                
                # Load image
                image = Image.open(uploaded_file)
                image_array = np.array(image)
                
                # Detect faces and emotions
                results = detect_faces_and_emotions(model, face_cascade, image_array)
                
                if results:
                    # Draw results
                    result_image = draw_emotion_results(image_array, results)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Original Image**")
                        st.image(image, use_column_width=True)
                    
                    with col2:
                        st.write("**Detection Results**")
                        st.image(result_image, use_column_width=True)
                    
                    # Display results
                    results_df = pd.DataFrame([
                        {
                            'Face': j+1,
                            'Emotion': result['emotion'],
                            'Confidence': f"{result['confidence']:.2%}"
                        }
                        for j, result in enumerate(results)
                    ])
                    st.dataframe(results_df, use_container_width=True)
                else:
                    st.warning(f"No faces detected in {uploaded_file.name}")
                
                st.markdown("---")
    
    with tab3:
        st.header("Model Information & Accuracy")
        
        # Model architecture
        st.subheader("Model Architecture")
        st.write("**Convolutional Neural Network (CNN)**")
        st.write("- Input: 48x48 grayscale images")
        st.write("- Architecture: 3 Conv2D layers + MaxPooling + Dense layers")
        st.write("- Output: 7 emotion classes")
        st.write("- Training Dataset: FER-2013 (35,887 images)")
        
        # Emotion classes
        st.subheader("Emotion Classes")
        emotion_info = pd.DataFrame({
            'Class ID': range(7),
            'Emotion': emotion_labels,
            'Description': [
                'Angry facial expression',
                'Disgusted facial expression',
                'Fearful facial expression',
                'Happy facial expression',
                'Sad facial expression',
                'Surprised facial expression',
                'Neutral facial expression'
            ]
        })
        st.dataframe(emotion_info, use_container_width=True)
        
        # Model performance (placeholder - you would load actual metrics)
        st.subheader("Model Performance")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Training Accuracy", "65.2%")
        
        with col2:
            st.metric("Validation Accuracy", "62.8%")
        
        with col3:
            st.metric("Test Accuracy", "61.5%")
        
        # Dataset information
        st.subheader("Dataset Information")
        st.write("**FER-2013 Dataset**")
        st.write("- Total Images: 35,887")
        st.write("- Image Size: 48x48 pixels")
        st.write("- Color: Grayscale")
        st.write("- Classes: 7 emotions")
        st.write("- Source: Facial Expression Recognition Challenge 2013")

if __name__ == "__main__":
    main()

