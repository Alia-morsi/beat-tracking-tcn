"""
Ben Hayes 2020

ECS7006P Music Informatics

Coursework 1: Beat Tracking

File: beat_tracking_tcn/datasets/ballroom_dataset.py
Description: A PyTorch dataset class representing the ballroom dataset.
"""

import torch
from torch.utils.data import Dataset

import os
import numpy as np
import glob


class ASAPDataset(Dataset):
    """
    A PyTorch Dataset wrapping the ballroom dataset for beat detection tasks

    Provides mel spectrograms and a vector of beat annotations per spectrogram
    frame as per the Davies & Böck paper (that is, with two frames of 0.5
    either side of the beat's peak.

    Requires dataset to be preprocessed to mel spectrograms using the provided
    script.
    """

    def __init__(
            self,
            spectrogram_dir,
            label_dir,
            sr=44100, #changed to fit the sr of the asap set
            hop_size_in_seconds=0.01,
            trim_size=(81, 3000),
            downbeats=False):
        """
        Initialise the dataset object.

        Parameters:
            spectrogram_dir: directory holding spectrograms as NumPy dumps
            label_dir: directory containing labels as NumPy dumps

        Keyword Arguments:
            sr (=22050): Sample rate to use when converting annotations to
                         spectrogram frame indices
            hop_size_in_seconds (=0.01): Mel spectrogram hop size
            trim_size (=(81,3000)): Dimensions to trim spectrogram down to.
                                    Should match input dimensions of network.
        """
        self.spectrogram_dir = spectrogram_dir
        self.label_dir = label_dir
        self.data_names = self._get_data_list()

        self.sr = sr
        self.hop_size = int(np.floor(hop_size_in_seconds * self.sr))
        self.trim_size = trim_size

        self.downbeats = downbeats
        self.meter = None #this should be set based on the meter lines in the asap dataset, so that the b and db strings can be changed
                          # to 1, 2, 3, 4 etc. 

    def __len__(self):
        """Overload len() calls on object."""
        return len(self.data_names)

    def __getitem__(self, i):
        """Overload square bracket indexing on object"""
        raw_spec, raw_beats = self._load_spectrogram_and_labels(i)
        x, y = self._trim_spec_and_labels(raw_spec, raw_beats)

        if self.downbeats:
            y = y.T

        return {
            'spectrogram': torch.from_numpy(
                    np.expand_dims(x.T, axis=0)).float(),
            'target': torch.from_numpy(y[:3000].astype('float64')).float(),
        }

    def get_name(self, i):
        """Fetches name of datapoint specified by index i"""
        return self.data_names[i]

    def get_ground_truth(self, i, quantised=True, downbeats=False):
        """
        Fetches ground truth annotations for datapoint specified by index i

        Parameters:
            i: Index signifying which datapoint to fetch truth for

        Keyword Arguments:
            quantised (=True): Whether to return a quantised grount truth
        """

        return self._get_quantised_ground_truth(i, downbeats)\
            if quantised else self._get_unquantised_ground_truth(i, downbeats)

    def _trim_spec_and_labels(self, spec, labels):
        """
        Trim spectrogram matrix and beat label vector to dimensions specified
        in self.trim_size. Returns tuple of trimmed NumPy arrays

        Parameters:
            spec: Spectrogram as NumPy array
            labels: Labels as NumPy array
        """

        x = np.zeros(self.trim_size)
        if not self.downbeats:
            y = np.zeros(self.trim_size[1])
        else:
            y = np.zeros((self.trim_size[1], 2))

        to_x = self.trim_size[0]
        to_y = min(self.trim_size[1], spec.shape[1])

        x[:to_x, :to_y] = spec[:, :to_y]
        y[:to_y] = labels[:to_y]

        return x, y

    def _get_data_list(self):
        """ Fetches list of datapoints in spectrogram directory.. Ben's code used the label directory, but in the asap case
        there are more labels than spectrograms """
        names = []
        #find all the leaves that have a ._annotation.text
        # their names would be formed of the whole path, which would be the same between the spec. and the label dirs. 
        # there might be more.txt files than spectrogram dirs because some of the asap files are only in midi (not audio) 
        annot_files = glob.glob('{}/**/*.npy'.format(self.spectrogram_dir), recursive=True)

        for annot_file in annot_files:
            relpath = os.path.relpath(annot_file, start=self.spectrogram_dir)
            names.append(os.path.splitext(relpath)[0])

        return names

    def _text_label_to_float(self, text):
        """Exracts beat time from a text line and converts to a float"""
        """ adapted to the asap dataset format """
        allowed = '1234567890. \t'
        #this can be done in a smarter way using the asap_annotations.json file, but for now let's keep it the same way
        #here we will just turn the db into 2 and the b into 1. not sure if this is a good way..
        
        filtered = ''.join([c for c in text if c in allowed])
        if '\t' in filtered:
            t = filtered.rstrip('\n').split('\t')
        else:
            t = filtered.rstrip('\n').split(' ')
        
        #remove extra info from keychange or timechange lines
        if ',' in t[2]:
            t[2] = t[2].split(',')[0]
            
        val = 1.0
        if t[2] == 'b':
            val = 2
        elif t[2] == 'db':
            val = 1
        else: #in case of bR, tho this numbering might not make sense at all..
            val = 3

        return float(t[1]), float(val) #t[0] and t[1] are time, and t[2] is db or b

    def _get_quantised_ground_truth(self, i, downbeats):
        """
        Fetches the ground truth (time labels) from the appropriate
        label file. Then, quantises it to the nearest spectrogram frames in
        order to allow fair performance evaluation.
        """
        with open(
                os.path.join(self.label_dir, self.data_names[i] + '_annotations.txt'),
                'r') as f:

            beat_times = []

            for line in f:
                time, index = self._text_label_to_float(line)
                if not downbeats:
                    beat_times.append(time * self.sr)
                else:
                    if index == 1:
                        beat_times.append(time * self.sr)
        quantised_times = []

        for time in beat_times:
            spec_frame = int(time / self.hop_size)
            quantised_time = spec_frame * self.hop_size / self.sr
            quantised_times.append(quantised_time)

        return np.array(quantised_times)

    def _get_unquantised_ground_truth(self, i, downbeats):
        """
        Fetches the ground truth (time labels) from the appropriate
        label file.
        """

        with open(
                os.path.join(self.label_dir, self.data_names[i] + '_annotations.txt'),
                'r') as f:
            
            beat_times = []

            for line in f:
                time, index = self._text_label_to_float(line)
                if not downbeats:
                    beat_times.append(time)
                else:
                    if index == 1:
                        beat_times.append(time)

        return np.array(beat_times)

    def _load_spectrogram_and_labels(self, i):
        """
        Given an index for the data name array, return the contents of the
        corresponding spectrogram and label dumps.
        """
        data_name = self.data_names[i]

        with open(
                os.path.join(self.label_dir, data_name + '_annotations.txt'),
                'r') as f:
            beat_floats = []
            beat_indices = []
            for line in f:
                parsed = self._text_label_to_float(line)
                beat_floats.append(parsed[0])
                beat_indices.append(parsed[1])
            beat_times = np.array(beat_floats) * self.sr

            if self.downbeats: #probably this will be working funny rn because of my gt scheme of 1, 0.5, and 2..
                downbeat_times = self.sr * np.array(
                    [t for t, i in zip(beat_floats, beat_indices) if i == 1])

        spectrogram =\
            np.load(os.path.join(self.spectrogram_dir, data_name + '.npy'))
        if not self.downbeats:
            beat_vector = np.zeros(spectrogram.shape[-1])
        else:
            beat_vector = np.zeros((spectrogram.shape[-1], 2))

        for time in beat_times:
            spec_frame =\
                min(int(time / self.hop_size), beat_vector.shape[0] - 1)
            for n in range(-2, 3): #what's this.. maybe this is an activation vector?
                if 0 <= spec_frame + n < beat_vector.shape[0]:
                    if not self.downbeats:
                        beat_vector[spec_frame + n] = 1.0 if n == 0 else 0.5
                    else:
                        beat_vector[spec_frame + n, 0] = 1.0 if n == 0 else 0.5
        
        if self.downbeats:
            for time in downbeat_times:
                spec_frame =\
                    min(int(time / self.hop_size), beat_vector.shape[0] - 1)
                for n in range(-2, 3):
                    if 0 <= spec_frame + n < beat_vector.shape[0]:
                        beat_vector[spec_frame + n, 1] = 1.0 if n == 0 else 0.5

        return spectrogram, beat_vector
