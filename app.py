from flask import Flask, render_template, request
import torch
import librosa
import numpy as np
import os
import webbrowser
import torch.nn.functional as F

from models.simple_cnn import CNNLSTM
from config import sample_rate, n_mels


app = Flask(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

model = CNNLSTM(num_classes=5).to(DEVICE)

model.load_state_dict(
    torch.load(
        "checkpoints/best_model.pth",
        map_location=DEVICE,
        weights_only=True
    )
)

model.eval()
CLASSES = [
    "COPD",            
    "Healthy",         
    "URTI",            
    "Bronchiectasis",  
    "Pneumonia"        
]
def extract_features(y, sr):
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
    feature = np.stack([logmel, mfcc], axis=0)

    return feature

def pad_or_crop(spec, max_len=300):
    if spec.dim() == 4:

        b, c, h, t = spec.shape

        if t < max_len:
            pad = max_len - t
            spec = F.pad(spec, (0, pad))
        else:
            spec = spec[:, :, :, :max_len]

    elif spec.dim() == 3:

        c, h, t = spec.shape

        if t < max_len:
            pad = max_len - t
            spec = F.pad(spec, (0, pad))
        else:
            spec = spec[:, :, :max_len]

    return spec

def preprocess_audio(file_path):

    y, sr = librosa.load(file_path, sr=sample_rate)
    feature = extract_features(y, sr)
    feature = torch.tensor(
        feature,
        dtype=torch.float32
    ).unsqueeze(0) 
    feature = pad_or_crop(feature, max_len=300)

    return feature.to(DEVICE)

@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None
    confidence = None
    probs_list = None


    if request.method == "POST":

        file = request.files.get("audio")

        if file:
            upload_dir = "uploads"
            os.makedirs(upload_dir, exist_ok=True)

            file_path = os.path.join(upload_dir, file.filename)

            file.save(file_path)

            x = preprocess_audio(file_path)

            with torch.no_grad():

                out = model(x)

                probs = torch.softmax(out, dim=1)

                pred = torch.argmax(probs, dim=1).item()

                prediction = CLASSES[pred]

                confidence = probs[0][pred].item() * 100

                probs_list = probs[0].cpu().numpy().tolist()


    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        probs=probs_list,
        classes=CLASSES
    )

if __name__ == "__main__":

    webbrowser.open("http://127.0.0.1:5000")

    app.run(debug=True)
