import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import librosa
import torch
from torch.utils.data import Dataset
import numpy as np
from collections import Counter
import torch.nn.functional as F
from scipy.signal import butter, lfilter
from config import sample_rate, n_mels, processed_path
CLASS_NAMES = [
    "Copd",
    "Healthy",
    "Urti",
    "Bronchiectasis",
    "Pneumonia"
]
CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASS_NAMES)}
def bandpass_filter(y, sr, low=100, high=2000):
    nyq = 0.5 * sr
    low = low / nyq
    high = high / nyq
    b, a = butter(5, [low, high], btype="band")
    return lfilter(b, a, y)
def preprocess_signal(y, sr):
    y = librosa.effects.preemphasis(y)
    y = bandpass_filter(y, sr)
    return y
def extract_features(y, sr):
    stft = np.abs(librosa.stft(y))
    stft_db = librosa.amplitude_to_db(stft, ref=np.max)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels
    )
    logmel = librosa.power_to_db(mel, ref=np.max)
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mels
    )
    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr
    )
    target_freq = n_mels
    target_time = logmel.shape[1]
    mfcc = librosa.util.fix_length(mfcc, size=target_time, axis=1)
    chroma = librosa.util.fix_length(chroma, size=target_time, axis=1)
    stft_db = librosa.util.fix_length(stft_db, size=target_time, axis=1)
    chroma = librosa.util.fix_length(chroma, size=target_freq, axis=0)
    stft_db = stft_db[:target_freq, :]
    mfcc = mfcc[:target_freq, :]
    feature = np.stack(
        [logmel, mfcc, chroma, stft_db],
        axis=0
    )
    return feature
def pad_or_crop(spec, max_len=300):
    c, h, t = spec.shape
    if t < max_len:
        pad = max_len - t
        spec = F.pad(spec, (0, pad))
    else:
        spec = spec[:, :, :max_len]
    return spec
class LungSoundDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        for cls in CLASS_NAMES:
            class_dir = os.path.join(root_dir, cls)
            label = CLASS_TO_IDX[cls]
            if not os.path.exists(class_dir):
                continue
            for file in os.listdir(class_dir):
                if file.endswith(".wav"):
                    self.samples.append(
                        (os.path.join(class_dir, file), label)
                    )
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        y, sr = librosa.load(path, sr=sample_rate)
        y = preprocess_signal(y, sr)
        feature = extract_features(y, sr)
        feature = torch.tensor(
            feature,
            dtype=torch.float32
        )
        feature = pad_or_crop(feature, max_len=300)

        label = torch.tensor(label, dtype=torch.long)

        return feature, label

def compute_class_weights(dataset):

    labels = [label for _, label in dataset.samples]

    counts = Counter(labels)

    total = sum(counts.values())

    num_classes = len(CLASS_NAMES)

    weights = torch.zeros(num_classes)

    for cls_idx, count in counts.items():

        weights[cls_idx] = total / (num_classes * count)

    return weights
