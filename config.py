sample_rate=22050
segmentation_length=4
n_mels=128

pitch_shift=[-1,1]
time_stretch=[0.95,1.05]
noise_std=0.005
gain_range=[0.95,1.05]
learning_rate = 3e-4
epoch = 80
batch_size = 16
focal_gamma=2

raw_audio_path=r"C:\Users\akhil\Desktop\respiration\data\raw_data"
processed_path=r"C:\Users\akhil\Desktop\respiration\data\processed_data"
csv_path=r"C:\Users\akhil\Desktop\respiration\data\patient_diagnosis.csv"
