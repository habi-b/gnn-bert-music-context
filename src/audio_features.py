import numpy as np
import librosa


def extract_mel_spec(audio_path, sr=22050, n_mels=128, fixed_frames=1300):

    y, _ = librosa.load(audio_path, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    if mel_db.shape[1] < fixed_frames:
        pad_width = fixed_frames - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode="constant",
                         constant_values=mel_db.min())
    else:
        mel_db = mel_db[:, :fixed_frames]

    return mel_db.astype(np.float32)
