from preprocessor import AudioPreprocessor
import os

import glob
import pandas as pd
import numpy as np

from PIL import Image
from tqdm import tqdm

def get_files(type):
    src_files = glob.glob(f"input/{type}/*.wav")
    dst_files = [os.path.split(file) for file in src_files]
    dst_files = [(path.split("/")+file.split(".")) for (path, file) in dst_files]
    dst_files = [(os.path.join("work",f[1],f"{f[2]}/ft.png"),
                os.path.join("work",f[1],f"{f[2]}/mel.png"),
                os.path.join("work",f[1],f"{f[2]}/cqt.png")) for f in dst_files]
    audio_files = [(src_files[i],dst_files[i]) for i in range(len(src_files))]
    
    return audio_files

audio_files = sorted(get_files("train_noisy"))

ap = AudioPreprocessor()
files = []
stft_means = []
stft_stds = []
mel_means = []
mel_stds = []
cqt_means = []
cqt_stds = []
for src,(stft_file, mel_file, cqt_file) in tqdm(audio_files):
    y = ap.read_audio(src)
    _, file = os.path.split(src)
    path, _ = os.path.split(stft_file)
    os.makedirs(path,exist_ok=True)
    stft = ap.normalize(ap.trim_blank(ap.audio_to_stft(y)))
    mel = ap.normalize(ap.trim_blank(ap.audio_to_mel(y)))
    cqt = ap.normalize(ap.trim_blank(ap.audio_to_cqt(y)))
    stft_mean, stft_std = ap.get_statistics(stft)
    mel_mean, mel_std = ap.get_statistics(mel)
    cqt_mean, cqt_std = ap.get_statistics(cqt)
    files.append(file)
    stft_means.append(stft_mean)
    stft_stds.append(stft_std)
    mel_means.append(mel_mean)
    mel_stds.append(mel_std)
    cqt_means.append(cqt_mean)
    cqt_stds.append(cqt_std)

stft_means = np.array(stft_means)
stft_stds = np.array(stft_stds)
mel_means = np.array(mel_means)
mel_stds = np.array(mel_stds)
cqt_means = np.array(cqt_means)
cqt_stds = np.array(cqt_stds)

df = pd.DataFrame({
    "file": files,
    **{f"stft_mean_{i}": stft_means[:, i] for i in range(stft_means.shape[1])},
    **{f"stft_std_{i}": stft_stds[:, i] for i in range(stft_stds.shape[1])},
    **{f"mel_mean_{i}": mel_means[:, i] for i in range(mel_means.shape[1])},
    **{f"mel_std_{i}": mel_stds[:, i] for i in range(mel_stds.shape[1])},
    **{f"cqt_mean_{i}": cqt_means[:, i] for i in range(cqt_means.shape[1])},
    **{f"cqt_std_{i}": cqt_stds[:, i] for i in range(cqt_stds.shape[1])},
})

print(df.shape)
df.to_csv("work/train_noisy_feature.csv", index=False)


