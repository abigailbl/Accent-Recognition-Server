from flask import Flask, jsonify, request, json
from keras._tf_keras.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

import librosa
import numpy as np
import joblib
import audioread
import wave
import subprocess
import audioread
import soundfile as sf

# Create the Flask application
app = Flask(__name__)

# Load the model and scaler
model = load_model('best_model3.keras')
scaler = joblib.load('scaler.save')

# Define the labels
labels = ['ENGLAND', 'INDIA', 'US']
le = LabelEncoder()
le.fit(labels)

# Function to extract features from a single audio file
def convert_to_wav(input_path, output_path):
    try:
        # Add the -y parameter to overwrite the file if it already exists
        subprocess.run(['ffmpeg', '-y', '-i', input_path, output_path], check=True)
        return True
    except Exception as e:
        print(f"Error converting file to WAV: {e}")
        return False

def check_audio_file(file_path):
    try:
        data, sr = sf.read(file_path)
        print(f"File loaded successfully with sample rate: {sr}")
        print(f"Number of samples: {len(data)}")
        return True
    except Exception as e:
        print(f"Error reading file: {e}")
        return False

# Call the function
check_audio_file('temp.wav')

def check_wave_file(file_path):
    try:
        with wave.open(file_path, 'rb') as wf:
            print(f"Channels: {wf.getnchannels()}")
            print(f"Sample width: {wf.getsampwidth()}")
            print(f"Frame rate: {wf.getframerate()}")
            print(f"Number of frames: {wf.getnframes()}")
            print(f"Duration: {wf.getnframes() / wf.getframerate()} seconds")
            return True
    except Exception as e:
        print(f"Error reading WAV file: {e}")
        return False

check_wave_file('temp.wav')

def extract_single_audio_features(file_path):
    try:
        # Load the audio using librosa
        audio, sr = librosa.load(file_path, sr=None)

        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)

        return mfccs_scaled
    except Exception as e:
        print(f"Error encountered while parsing file: {file_path}\n{e}")
        return None

@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Get the data from the request
        data = request.files['file']

        # Save the temporary file
        file_path = 'temp.wav'
        data.save(file_path)

        temp_output_path = 'C:/Users/USER/Desktop/Speech-Accent-Recognition-master/src/outputfile.wav'
        if not convert_to_wav(file_path, temp_output_path):
            return "File conversion failed", 500

        # Extract features from the file
        features = extract_single_audio_features(temp_output_path)

        if features is not None:
            # Normalize the features
            features_scaled = scaler.transform([features])

            # Reshape the features to fit the input shape of the model
            features_reshaped = features_scaled.reshape(1, features_scaled.shape[1], 1, 1)

            # Predict the accent
            prediction = model.predict(features_reshaped)
            predicted_class = np.argmax(prediction, axis=1)

            # Convert the result to the language name
            predicted_language = le.inverse_transform(predicted_class)

            # Return the result as JSON
            return jsonify(predicted_language[0])
        else:
            return jsonify({'error': 'Failed to extract features from the audio file.'}), 400
    except Exception as e:
        print(f"Exception in predict route: {e}")  # Print the error details
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
