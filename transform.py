from preprocessor import AudioPreprocessor
import os

import glob
import pandas as pd

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

audio_files = get_files("trn_curated")
# print(audio_files[:1000])

# files = pd.read_csv("input/train_curated.csv")
# print(list(files["fname"].iloc[:1000]))

ap = AudioPreprocessor()
for src,(stft_file, mel_file, cqt_file) in tqdm(audio_files):
    y = ap.read_audio(src)
    path, file = os.path.split(stft_file)
    os.makedirs(path,exist_ok=True)
    # stft = ap.audio_to_stft(y)
    mel = ap.audio_to_mel(y)
    # cqt = ap.audio_to_cqt(y)
    ap.to_image(mel, mode="rgb").save(mel_file)
    #break
