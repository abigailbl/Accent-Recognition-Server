import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from keras._tf_keras.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from keras._tf_keras.keras.utils import to_categorical
#from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras._tf_keras.keras.callbacks import EarlyStopping, ModelCheckpoint
#from tensorflow.keras.layers import EarlyStopping, ModelCheckpoint
import joblib

# Directory where audio files are saved
audio_dir = 'audio'


# Function to augment audio
def augment_audio(audio, sample_rate):
    augmented = []

    # Original
    augmented.append(audio)

    # Noise injection
    noise = np.random.randn(len(audio))
    augmented.append(audio + 0.005 * noise)

    # Shifting time
    augmented.append(np.roll(audio, int(sample_rate / 10)))

    # Changing pitch
    augmented.append(librosa.effects.pitch_shift(y=audio, sr=sample_rate, n_steps=4))
    augmented.append(librosa.effects.pitch_shift(y=audio, sr=sample_rate, n_steps=-4))

    # Time-stretching
    augmented.append(librosa.effects.time_stretch(y=audio, rate=1.25))
    augmented.append(librosa.effects.time_stretch(y=audio, rate=0.8))

    return augmented

# Function to extract audio features
def extract_features(file_path):
    try:
        print(f"Processing file: {file_path}")
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        augmented_audios = augment_audio(audio, sample_rate)
        features = []
        for aug_audio in augmented_audios:
            mfccs = librosa.feature.mfcc(y=aug_audio, sr=sample_rate, n_mfcc=40)
            mfccs_scaled = np.mean(mfccs.T, axis=0)
            features.append(mfccs_scaled)
        return features
    except Exception as e:
        print(f"Error encountered while parsing file: {file_path}\n{e}")
        return None

# Extract features and labels
features = []
labels = []

# Iterate over all files in the audio directory
for file_name in os.listdir(audio_dir):
    if file_name.endswith('.wav'):
        file_path = os.path.join(audio_dir, file_name)
        print(f"Extracting features from {file_path}")
        # Extract native language from filename by removing trailing numbers and .wav extension
        native_language = ''.join([char for char in file_name if not char.isdigit()]).replace('.wav', '')

        mfccs_scaled = extract_features(file_path)
        if mfccs_scaled is not None:
            features.extend(mfccs_scaled)  # Note the use of extend instead of append
            labels.extend([native_language] * len(mfccs_scaled))  # Extend labels to match augmented features !!!!!!!

print("Finished extracting features")

# Check if any features were extracted
if len(features) == 0:
    raise ValueError("No audio features were extracted. Please check the file paths and ensure the audio files are accessible.")

# Convert into numpy arrays
X = np.array(features)
y = np.array(labels)

# Encode the labels
le = LabelEncoder()
yy = le.fit_transform(y)

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, 'scaler.save')


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, yy, test_size=0.2, random_state=42)

# Convert labels to categorical
num_classes = len(np.unique(y_train))
y_train = to_categorical(y_train, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)

# Reshape features to fit CNN input shape
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1, 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1, 1)

# Define the CNN model
model = Sequential()
model.add(Input(shape=(40, 1, 1)))
model.add(Conv2D(64, kernel_size=(3, 1), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Dropout(0.3))
model.add(Conv2D(128, kernel_size=(3, 1), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Dropout(0.3))
model.add(Conv2D(256, kernel_size=(3, 1), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


print(model.summary())


# Train the model with early stopping and model checkpointing
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('best_model3.keras', save_best_only=True, monitor='val_loss', mode='min')
]


model.fit(X_train, y_train, epochs=70, batch_size=32, validation_data=(X_test, y_test), callbacks=callbacks)

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print("Accuracy:", accuracy_score(y_true, y_pred_classes))
print(classification_report(y_true, y_pred_classes, target_names=le.classes_))

# Save the trained model to a file
model.save('best_model3.keras')

