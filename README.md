# Accent Recognition
This project is an Accent Recognition system that classifies spoken English accents into three categories: England, India, and US. It includes a Convolutional Neural Network (CNN) model for classifying accents from audio files and a Flask API server for real-time predictions.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Model Training](#model-training)
- [Flask Server](#flask-server)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
  - [Model Training](#model-training-usage)
  - [Running the Flask Server](#running-the-flask-server)
  - [Making Predictions](#making-predictions)
- [Requirements](#requirements)
- [Notes](#notes)
## Overview
The purpose of this project is to train a model capable of recognizing English accents based on audio samples. This project includes:
- A Machine Learning model implemented in Python using Keras and TensorFlow to train a CNN for accent recognition.
- A Flask API that allows users to upload audio files and receive predictions of the accent category.

## Project Structure
- `audio/`: Directory containing audio files for training.
- `best_model3.keras`: Saved CNN model for predictions.
- `scaler.save`: Saved scaler for feature normalization.
- `flask_server.py`: Flask API server to handle audio file uploads and provide predictions.

## Model Training
The accent recognition model uses a CNN to classify audio features derived from MFCCs (Mel-frequency cepstral coefficients). The following steps are performed during model training:

1. **Data Augmentation**: Various techniques like noise injection, pitch shifting, and time-stretching are applied.
2. **Feature Extraction**: MFCC features are extracted from each audio file.
3. **Model Architecture**: The CNN model is defined with convolutional layers, max-pooling, dropout, and fully connected layers.
4. **Training**: The model is trained on augmented audio features with early stopping and model checkpointing.
5. **Evaluation**: The trained model's performance is evaluated using accuracy and a classification report.

## Flask Server
The Flask server (`flask_server.py`) provides a REST API to predict accents. Users can upload audio files to the `/predict` endpoint, where the server will:
1. Convert the audio file to WAV format if needed.
2. Extract MFCC features.
3. Scale and reshape the features for the model input.
4. Use the trained model to predict the accent and return the result in JSON format.

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository_url>
   cd <repository_name>
2. **Install required packages**
Make sure you have Python 3.x installed, then install the required packages:

   ```bash 
   pip install -r requirements.txt
3. **Place Audio Files**
Place the .wav audio files for training in the audio/ directory.

4. **Model Training**
If you need to retrain the model, run the MODEL script section in the code provided above. The trained model will be saved as best_model3.keras, and the scaler will be saved as scaler.save.

## Usage
### Model Training Usage
To train the model:

1. Ensure audio files are located in the audio/ directory.
2. Run the MODEL script to train and save the model:
   ```bash 
   python model_training.py
3. The trained model and scaler will be saved in the current directory.
### Running the Flask Server
Start the Flask server:

     ```bash
     flask run --host=0.0.0.0
The server will start on http://0.0.0.0:5000.

### Making Predictions
To predict an accent, send a POST request to the /predict endpoint:
- **URL:** http://localhost:5000/predict
- **Method:** POST
- **Body:** Form-data with a key "file" and the audio file in WAV format.
- **Response:** JSON object with the predicted accent.


## Requirements
- Python 3.x
- Libraries: librosa, scikit-learn, keras, tensorflow, flask, joblib, audioread, soundfile
- ffmpeg (for audio file conversion)

Install additional libraries using:

    ```bash
    pip install -r requirements.txt


## Notes
- Ensure that ffmpeg is installed and accessible from the command line for audio format conversion.
- Modify audio_dir or other paths as needed.
- The model currently recognizes only three accents. You can extend the dataset and retrain to add more accents if required.
