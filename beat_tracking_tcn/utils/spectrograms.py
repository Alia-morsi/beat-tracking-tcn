"""
Ben Hayes 2020

ECS7006P Music Informatics

Coursework 1: Beat Tracking

File: beat_tracking_tcn/utils/spectrograms.py

Descrption: Utility functions for computing and trimming mel spectrograms.
"""
import os

import librosa
import numpy as np

def create_spectrogram(
        file_path,
        n_fft,
        hop_length_in_seconds,
        n_mels):
    
    x, sr = librosa.load(file_path)
    hop_length_in_samples = int(np.floor(hop_length_in_seconds * sr))
    spec = librosa.feature.melspectrogram(
        y=x,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length_in_samples,
        n_mels=n_mels)
    mag_spec = np.abs(spec)

    return mag_spec

def create_spectrograms(
        audio_dir,
        audio_file,
        spectrogram_dir,
        n_fft,
        hop_length_in_seconds,
        n_mels):

    mag_spec = create_spectrogram(
        audio_file,
        n_fft,
        hop_length_in_seconds,
        n_mels)
    
    #file root has to be in the path of audio file, bec accordingly we calculate the dir structure in out
    #make check for above case
    relpath = os.path.relpath(audio_file, start=audio_dir)
    os.makedirs(os.path.join(spectrogram_dir, os.path.dirname(relpath)), exist_ok=True)
    np.save(os.path.join(spectrogram_dir, os.path.splitext(relpath)[0]), mag_spec)
    print('Saved spectrum for {}'.format(os.path.join(spectrogram_dir, os.path.splitext(relpath)[0])))

def trim_spectrogram(spectrogram, trim_size):
    output = np.zeros(trim_size)
    dim0_range = min(trim_size[0], spectrogram.shape[0])
    dim1_range = min(trim_size[1], spectrogram.shape[1])

    output[:dim0_range, :dim1_range] = spectrogram[:dim0_range, :dim1_range]
    return output
