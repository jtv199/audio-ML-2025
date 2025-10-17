import librosa
import numpy as np

from PIL import Image


class AudioPreprocessor:
    def __init__(
        self,
        sampling_rate=44100,
        duration=2,
        hop_length=512,
        fmin=20,
        n_mels=128,
        n_fft=2048,
        ref=np.max,
    ):
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = sampling_rate // 2
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.samples = sampling_rate * duration
        self.ref = ref

    def read_audio(self, pathname, trim_long_data=False):
        y, _ = librosa.load(pathname, sr=self.sampling_rate)
        # trim silence
        if 0 < len(y):  # workaround: 0 length causes error
            y, _ = librosa.effects.trim(y)  # trim, top_db=default(60)
        # make it unified length to conf.samples
        if len(y) > self.samples:  # long enough
            if trim_long_data:
                y = y[0 : 0 + self.samples]
        else:  # pad blank
            padding = self.samples - len(y)  # add padding at both ends
            offset = padding // 2
            y = np.pad(y, (offset, self.samples - len(y) - offset), "constant")
        return y

    def audio_to_stft(self, audio):
        spectrogram = librosa.stft(
            y=audio,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
        )
        spectrogram = librosa.amplitude_to_db(abs(spectrogram), ref=self.ref)
        spectrogram = spectrogram.astype(np.float32)
        if self.ref == np.max:
            spectrogram = np.clip(spectrogram, -80, 0)
        return spectrogram
    
    def audio_to_cqt(self, audio):
        spectrogram = librosa.cqt(
            y=audio,
            sr=self.sampling_rate,
            hop_length=self.hop_length,
        )
        spectrogram = librosa.amplitude_to_db(abs(spectrogram),ref=self.ref)
        spectrogram = spectrogram.astype(np.float32)
        if self.ref == np.max:
            spectrogram = np.clip(spectrogram, -80, 0)
        return spectrogram

    def audio_to_mel(self, audio):
        spectrogram = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sampling_rate,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            fmin=self.fmin,
            fmax=self.fmax,
        )
        spectrogram = librosa.power_to_db(spectrogram,ref=self.ref)
        spectrogram = spectrogram.astype(np.float32)
        if self.ref == np.max:
            spectrogram = np.clip(spectrogram, -80, 0)
        return spectrogram
    
    def trim_blank(self, spectrogram, threshhold=1e-6):
        maximum = np.max(spectrogram)
        sum = np.sum(spectrogram + (80 - maximum), axis=0)
        mask = np.abs(sum) > threshhold
        if not mask.any():
            return spectrogram
        i0 = int(mask.argmax())
        i1 = len(sum) - int(mask[::-1].argmax()) - 1
        trimmed_spectrogram = spectrogram[:,i0:i1+1]
        return trimmed_spectrogram
    
    def get_statistics(self, spectrogram):
        # print(spectrogram.shape)
        maximum = np.max(spectrogram)
        sum = np.sum(spectrogram + (80 - maximum), axis=0)
        if sum[0] < 1e-6:
            spectrogram = self.trim_blank(spectrogram)
        # print(spectrogram.shape)
        mean = np.mean(spectrogram,axis=1)
        std = np.std(spectrogram,axis=1)
        return mean, std
    
    def normalize(self, spectrogram):
        mean = spectrogram.mean()
        std = spectrogram.std()
        spectrogram = (spectrogram - mean) / (std + 1e-6)
        norm_max = spectrogram.max()
        norm_min = spectrogram.min()
        spectrogram = (spectrogram - norm_min) / (norm_max - norm_min)
        return spectrogram
    
    def to_image(self, spectrogram, mode="mono", normalize=True):
        if normalize:
            spectrogram = self.normalize(spectrogram)
        if mode == "mono":
            spectrogram = 65535 * spectrogram
            spectrogram = spectrogram.astype(np.uint16)
        elif mode == "rgb":
            spectrogram = 255 * spectrogram
            spectrogram = np.stack([spectrogram for _ in range(3)], axis=-1)
            spectrogram = spectrogram.astype(np.uint8)
        img = Image.fromarray(spectrogram)
        return img
    
    def to_spectrogram(self, img):
        spectrogram = np.array(img).astype(np.float32)
        spectrogram = (spectrogram / 65535 - 1) * 80
        return spectrogram

if __name__ == "__main__":
    sample = "input/trn_curated/1d44b0bd.wav"
    # sample = "input/trn_curated/0a9f7b92.wav"
    ap = AudioPreprocessor()
    audio = ap.read_audio(sample)
    mel = ap.audio_to_mel(audio)
    mel_t = ap.trim_blank(mel)
    # print(mel_t)
    # print(mel.shape)
    # print(mel_t.shape)
    mel_img = ap.to_image(mel)
    mel_img.show()
    mel_rimg = ap.to_image(mel_t)
    mel_rimg.show()
    mean, std = ap.get_statistics(mel)
    print(mean, std)

### [[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8]]