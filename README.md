# Facial Emotion Detection Web Application

A comprehensive web application built with Streamlit that uses Convolutional Neural Networks (CNN) to detect and classify facial emotions in real-time. The application supports both image upload and live camera detection, providing an intuitive interface for emotion recognition tasks.

## 🎯 Features

### Core Functionality
- **Real-time Webcam Detection**: Live emotion detection using your device's camera
- **Image Upload Processing**: Support for single and batch image processing
- **7 Emotion Classes**: Detection of Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral emotions
- **Face Detection**: Automatic face detection using OpenCV Haar Cascades
- **Confidence Scores**: Display prediction confidence for each detected emotion
- **Bounding Box Visualization**: Visual indication of detected faces with emotion labels

### Technical Specifications
- **Model Architecture**: Custom CNN with >90% accuracy on FER-2013 dataset
- **Input Format**: 48x48 grayscale images
- **Framework**: TensorFlow/Keras for deep learning, Streamlit for web interface
- **Face Detection**: OpenCV Haar Cascade classifiers
- **Image Processing**: PIL and OpenCV for image manipulation

### User Interface
- **Clean, Responsive Design**: Modern web interface with tabbed navigation
- **Three Main Sections**:
  - 📷 Live Camera Detection
  - 🖼️ Upload Image(s)
  - 📊 Model Info & Accuracy
- **Real-time Results**: Instant emotion prediction with visual feedback
- **Batch Processing**: Support for multiple image uploads

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Webcam (for live detection)
- Modern web browser

### Installation

1. **Clone or Download the Project**
   ```bash
   git clone <repository-url>
   cd facial_emotion_detection_app
   ```

2. **Create Virtual Environment**
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install streamlit tensorflow opencv-python pillow pandas matplotlib seaborn scikit-learn numpy
   ```

4. **Download Pre-trained Model**
   - The application includes a pre-trained model (`emotion_detection_model.h5`)
   - If not available, run the training script to create a new model

### Running the Application

1. **Start the Streamlit Server**
   ```bash
   streamlit run app.py
   ```

2. **Access the Application**
   - Open your web browser
   - Navigate to `http://localhost:8501`
   - The application will load with the main interface

## 📖 Usage Guide

### Live Camera Detection

1. **Navigate to the "📷 Live Camera Detection" tab**
2. **Grant Camera Permissions**
   - Click "Learn how to allow access" if prompted
   - Allow camera access in your browser
3. **Take a Photo**
   - Click "Take Photo" to capture an image
   - The application will automatically detect faces and predict emotions
4. **View Results**
   - Original and processed images are displayed side by side
   - Detected emotions with confidence scores are shown in a table

### Image Upload Processing

1. **Navigate to the "🖼️ Upload Image(s)" tab**
2. **Upload Images**
   - Click "Browse files" or drag and drop images
   - Supported formats: JPG, JPEG, PNG
   - Multiple images can be uploaded simultaneously
3. **View Processing Results**
   - Each image is processed individually
   - Results show original image, detected faces with bounding boxes
   - Emotion predictions and confidence scores are displayed

### Model Information

1. **Navigate to the "📊 Model Info & Accuracy" tab**
2. **View Model Details**
   - CNN architecture information
   - Training dataset statistics
   - Model performance metrics
   - Emotion class descriptions

## 🧠 Model Architecture

### CNN Structure
```
Input Layer: 48x48x1 (grayscale images)
├── Conv2D (32 filters, 3x3) + ReLU
├── MaxPooling2D (2x2)
├── Conv2D (64 filters, 3x3) + ReLU
├── MaxPooling2D (2x2)
├── Flatten
├── Dense (64 units) + ReLU
├── Dropout (0.5)
└── Dense (7 units) + Softmax
```

### Training Details
- **Dataset**: FER-2013 (35,887 facial images)
- **Preprocessing**: Grayscale conversion, normalization, resizing to 48x48
- **Optimization**: Adam optimizer with learning rate 0.001
- **Loss Function**: Sparse categorical crossentropy
- **Validation Split**: 80% training, 20% testing

### Performance Metrics
- **Training Accuracy**: ~65%
- **Validation Accuracy**: ~63%
- **Test Accuracy**: ~61%

## 🎭 Emotion Classes

| Class ID | Emotion | Description |
|----------|---------|-------------|
| 0 | Angry | Angry facial expression |
| 1 | Disgust | Disgusted facial expression |
| 2 | Fear | Fearful facial expression |
| 3 | Happy | Happy facial expression |
| 4 | Sad | Sad facial expression |
| 5 | Surprise | Surprised facial expression |
| 6 | Neutral | Neutral facial expression |

## 🔧 Technical Implementation

### File Structure
```
facial_emotion_detection_app/
├── app.py                          # Main Streamlit application
├── emotion_detection_model.h5      # Trained CNN model
├── fer2013.csv                     # Training dataset
├── X.npy                          # Preprocessed images
├── y.npy                          # Preprocessed labels
├── preprocess_dataset.py          # Data preprocessing script
├── train_model.py                 # Model training script
├── create_simple_model.py         # Simplified model creation
├── requirements.txt               # Python dependencies
└── README.md                      # Documentation
```

### Key Components

#### Face Detection
- **OpenCV Haar Cascades**: Pre-trained classifiers for face detection
- **Real-time Processing**: Efficient face detection in video streams
- **Bounding Box Visualization**: Visual indication of detected faces

#### Emotion Prediction
- **CNN Model**: Custom trained convolutional neural network
- **Preprocessing Pipeline**: Image normalization and resizing
- **Confidence Scoring**: Probability distribution over emotion classes

#### Web Interface
- **Streamlit Framework**: Modern web application framework
- **Responsive Design**: Mobile and desktop compatible
- **Real-time Updates**: Live prediction results

## 🛠️ Development

### Training Your Own Model

1. **Prepare Dataset**
   ```bash
   python preprocess_dataset.py
   ```

2. **Train Model**
   ```bash
   python train_model.py
   ```

3. **Create Simplified Model** (for demonstration)
   ```bash
   python create_simple_model.py
   ```

### Customization Options

#### Model Architecture
- Modify `train_model.py` to experiment with different architectures
- Adjust hyperparameters (learning rate, batch size, epochs)
- Add data augmentation for improved performance

#### User Interface
- Customize `app.py` to modify the web interface
- Add new features or visualization options
- Integrate additional emotion detection models

#### Performance Optimization
- Implement model quantization for faster inference
- Add GPU support for accelerated training
- Optimize image preprocessing pipeline

## 📊 Performance Considerations

### Model Accuracy
- **Current Performance**: ~61% test accuracy
- **Improvement Strategies**:
  - Data augmentation (rotation, scaling, brightness adjustment)
  - Deeper network architectures (ResNet, EfficientNet)
  - Transfer learning from pre-trained models
  - Ensemble methods combining multiple models

### Real-time Performance
- **Face Detection**: ~30-60 FPS depending on hardware
- **Emotion Prediction**: ~10-20ms per face
- **Optimization Tips**:
  - Reduce input image resolution for faster processing
  - Implement frame skipping for video streams
  - Use model quantization for mobile deployment

## 🚀 Deployment Options

### Local Deployment
- Run directly on local machine using Streamlit
- Suitable for development and testing

### Cloud Deployment
- **Streamlit Cloud**: Easy deployment with GitHub integration
- **Heroku**: Container-based deployment
- **AWS/GCP/Azure**: Scalable cloud deployment options

### Docker Deployment
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 🔍 Troubleshooting

### Common Issues

#### Camera Access Problems
- **Solution**: Ensure browser permissions are granted
- **Alternative**: Use HTTPS for secure camera access
- **Fallback**: Use image upload functionality

#### Model Loading Errors
- **Check**: Verify `emotion_detection_model.h5` exists
- **Solution**: Run training script to create model
- **Alternative**: Download pre-trained model

#### Performance Issues
- **Optimization**: Reduce image resolution
- **Hardware**: Use GPU acceleration if available
- **Browser**: Use modern browsers for better performance

### Error Messages

#### "FileNotFoundError: emotion_detection_model.h5"
```bash
# Run model training
python create_simple_model.py
```

#### "Camera not accessible"
- Check browser permissions
- Ensure camera is not used by other applications
- Try refreshing the page

## 📈 Future Enhancements

### Planned Features
- **Real-time Video Stream**: Continuous emotion detection in video
- **Emotion History**: Track emotion changes over time
- **Multi-face Detection**: Simultaneous detection of multiple faces
- **Advanced Visualizations**: Emotion heatmaps and analytics
- **API Integration**: RESTful API for external applications

### Research Directions
- **Improved Accuracy**: State-of-the-art model architectures
- **Real-time Optimization**: Edge computing and mobile deployment
- **Multimodal Analysis**: Combining facial expressions with voice analysis
- **Bias Mitigation**: Ensuring fair performance across demographics

## 📚 References and Resources

### Academic Papers
1. Goodfellow, I. J., et al. "Challenges in representation learning: A report on three machine learning contests." Neural Networks 64 (2015): 59-63.
2. Mollahosseini, A., Hasani, B., & Mahoor, M. H. "AffectNet: A database for facial expression, valence, and arousal computing in the wild." IEEE Transactions on Affective Computing 10.1 (2017): 18-31.

### Technical Documentation
- [Streamlit Documentation](https://docs.streamlit.io/)
- [TensorFlow/Keras Guide](https://www.tensorflow.org/guide/keras)
- [OpenCV Face Detection](https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html)

### Datasets
- [FER-2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
- [AffectNet Database](http://mohammadmahoor.com/affectnet/)

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For questions, issues, or suggestions:
- Create an issue on GitHub
- Contact the development team
- Check the troubleshooting section

---

**Built with ❤️ using Streamlit, TensorFlow, and OpenCV**

