import os
import sys
import librosa
import soundfile as sf

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from annotation_parser import parse_annotation
from disease_map import load_disease_map, get_patient_id
from config import sample_rate, raw_audio_path, processed_path

disease_csv=r"C:\Users\akhil\Desktop\respiration\data\patient_diagnosis.csv"

def segment_all():
    disease_map=load_disease_map(disease_csv)
    for disease in set(disease_map.values()):
        os.makedirs(os.path.join(processed_path,disease),exist_ok=True)
    for file in os.listdir(raw_audio_path):
        if not file.endswith(".txt"):
            continue
        txt_path=os.path.join(raw_audio_path,file)
        wav_path=txt_path.replace(".txt",".wav")
        if not os.path.exists(wav_path):
            continue
        patient_id=get_patient_id(file)
        disease=disease_map.get(patient_id)
        if disease is None:
            print(f" No disease label for patient {patient_id}")
            continue
        annotations=parse_annotation(txt_path)
        y,sr=librosa.load(wav_path,sr=sample_rate)
        base=os.path.splitext(file)[0]

        for i, ann in enumerate(annotations):
            start=int(ann["start"]*sr)
            end = int(ann["end"] * sr)
            segment = y[start:end]
            out_name = f"{base}_seg{i}.wav"
            out_path = os.path.join(processed_path, disease, out_name)
            sf.write(out_path, segment, sr)
        print(f"Processed {base} → {disease}")
if __name__ == "__main__":
    segment_all()