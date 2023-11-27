
import h5py
import os
from tqdm import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import IPython.display as ipd
import h5py
import tensorflow as tf
import keras
# from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.sequence import pad_sequences


def extract_mel_spectrogram(audio, sr, window_size=25, hop_size=10, n_mels=80):
    # Compute mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=int(sr * window_size / 1000), hop_length=int(sr * hop_size / 1000), n_mels=n_mels)

    return librosa.power_to_db(mel_spectrogram, ref=np.max)


def mean_variance_normalization(mel_spectrogram):
    # Mean-variance normalization
    mean = np.mean(mel_spectrogram)
    std = np.std(mel_spectrogram)
    normalized_mel = (mel_spectrogram - mean) / std

    return normalized_mel

def frame_level_normalization(mel_spectrogram):
    # Frame-level normalization
    mean_per_frame = np.mean(mel_spectrogram, axis=1, keepdims=True)
    std_per_frame = np.std(mel_spectrogram, axis=1, keepdims=True)

    normalized_mel = (mel_spectrogram - mean_per_frame) / std_per_frame

    return normalized_mel
def trim_silence(audio_path, threshold=40):
    audio = None  # Initialize audio to None
    try:
        # Load the audio file
        audio, sr = librosa.load(audio_path, sr=44100)

        # Print the samplerate for debugging
        # print("Samplerate:", sr)

        # Trim silence
        trimmed_audio, _ = librosa.effects.trim(audio, top_db=threshold)

        return trimmed_audio
    except Exception as e:
        print(f"Error parocessing {audio_path}: {str(e)}")
        return audio


# Define the path to your dataset
dataset_path = '/data/common_source/datasets/vctk-corpus/VCTK-Corpus/VCTK-Corpus/wav48'

# Create a list to store spectrograms
spectrograms = []

# Inside the loop where you process files for each speaker
for folder in tqdm(os.listdir(dataset_path)[:5], desc="Processing folders"):
    folder_path = os.path.join(dataset_path, folder)
    if os.path.isdir(folder_path):
        speaker_spectrograms = []  # List to store spectrograms for the current speaker
        for file in tqdm(os.listdir(folder_path), desc=f"Processing files in {folder}"):
            file_path = os.path.join(folder_path, file)
            trimmed_audio = trim_silence(file_path, threshold=40)

            if trimmed_audio is not None:
                mel_spectrogram = extract_mel_spectrogram(trimmed_audio, sr=44100, window_size=25, hop_size=10, n_mels=80)
                normalized_mel = mean_variance_normalization(mel_spectrogram)
                frame_normalized_mel = frame_level_normalization(normalized_mel)
                speaker_spectrograms.append(frame_normalized_mel)

        # Check the lengths of all sequences before padding
        lengths_before_padding = [spec.shape[1] for spec in speaker_spectrograms]
        print(f"Before padding - Max length: {max(lengths_before_padding)}, Shapes: {lengths_before_padding}")

        try:
            # Transpose the spectrograms before padding
            transposed_spectrograms = [spec.T for spec in speaker_spectrograms]

            # Pad the spectrograms for the current speaker to the maximum length per batch
            max_length = max(lengths_before_padding)
            padded_spectrogram = pad_sequences(transposed_spectrograms, maxlen=max_length, padding='post', dtype='float32', truncating='post', value=None)

            # Print the shapes after padding
            print(f"After padding - Max length: {max_length}, Padded shape: {padded_spectrogram.shape}")

            # Append the padded spectrograms to the list
            spectrograms.append(padded_spectrogram)
        except ValueError as ve:
            print(f"Error in padding: {ve}")
            # Print the shape of each sequence causing the error
            for i, spec in enumerate(transposed_spectrograms):
                print(f"Sample {i + 1} - Shape: {spec.shape}")

# Create an HDF5 file to save the processed spectrograms
output_file = 'processed_spectrograms_test.h5'
with h5py.File(output_file, 'w') as hf:
    for folder, padded_spectrogram in zip(os.listdir(dataset_path), spectrograms):
        # Assuming the folder name is the speaker ID
        speaker_id = folder

        # Save the padded spectrograms in the HDF5 file
        if speaker_id in hf:
            hf[speaker_id].resize((hf[speaker_id].shape[0] + 1, *padded_spectrogram.shape[1:]))
            hf[speaker_id][-1] = padded_spectrogram
        else:
            hf.create_dataset(speaker_id, data=padded_spectrogram, maxshape=(None, *padded_spectrogram.shape[1:]), chunks=True)