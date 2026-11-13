"""Загрузка аудио и аугментация для третьей лабораторной работы."""

from pathlib import Path
import random

import numpy as np
import soundfile as sf
from scipy import signal

from constants import (
    ERROR_AUDIO_EMPTY,
    ERROR_NOISE_FILES_MISSING,
    ERROR_RIR_FILES_MISSING,
    NOISE_CATEGORY_MUSIC,
    NOISE_CATEGORY_NOISE,
    NOISE_CATEGORY_SPEECH,
)


def loadWAV(filename, max_frames, evalmode=True, num_eval=10):
    """Загрузить один или несколько фрагментов WAV фиксированной длины."""
    maximum_audio_length = max_frames * 160 + 240
    audio, _ = sf.read(filename, always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.size == 0:
        raise ValueError(ERROR_AUDIO_EMPTY.format(path=filename))

    if audio.size <= maximum_audio_length:
        shortage = maximum_audio_length - audio.size + 1
        audio = np.pad(audio, (0, shortage), mode="wrap")

    if evalmode:
        start_frames = np.linspace(
            0,
            audio.size - maximum_audio_length,
            num=num_eval,
        )
    else:
        start_frames = np.array(
            [random.random() * (audio.size - maximum_audio_length)]
        )

    if evalmode and max_frames == 0:
        fragments = [audio]
    else:
        fragments = [
            audio[int(start) : int(start) + maximum_audio_length]
            for start in start_frames
        ]
    return np.stack(fragments).astype(np.float32)


class AugmentWAV:
    """Онлайн-аугментация шумами MUSAN и импульсными откликами RIR."""

    def __init__(self, musan_path, rir_path, max_frames):
        self.max_frames = max_frames
        self.max_audio = max_frames * 160 + 240
        self.noise_types = (
            NOISE_CATEGORY_NOISE,
            NOISE_CATEGORY_SPEECH,
            NOISE_CATEGORY_MUSIC,
        )
        self.noise_snr = {
            NOISE_CATEGORY_NOISE: (0, 15),
            NOISE_CATEGORY_SPEECH: (13, 20),
            NOISE_CATEGORY_MUSIC: (5, 15),
        }
        self.noise_count = {
            NOISE_CATEGORY_NOISE: (1, 1),
            NOISE_CATEGORY_SPEECH: (3, 7),
            NOISE_CATEGORY_MUSIC: (1, 1),
        }
        self.noise_files = {noise_type: [] for noise_type in self.noise_types}
        for file_path in Path(musan_path).rglob("*.wav"):
            for noise_type in self.noise_types:
                if noise_type in file_path.parts:
                    self.noise_files[noise_type].append(file_path)
                    break
        self.rir_files = list(Path(rir_path).rglob("*.wav"))

    def additive_noise(self, noise_category, audio):
        """Добавить шум заданной категории с случайным отношением сигнал/шум."""
        available_files = self.noise_files.get(noise_category, [])
        if not available_files:
            raise FileNotFoundError(
                ERROR_NOISE_FILES_MISSING.format(category=noise_category)
            )
        clean_db = 10.0 * np.log10(np.mean(audio**2) + 1e-4)
        minimum_count, maximum_count = self.noise_count[noise_category]
        selected_count = min(
            random.randint(minimum_count, maximum_count),
            len(available_files),
        )
        selected_files = random.sample(available_files, selected_count)
        noises = []
        for noise_file in selected_files:
            noise_audio = loadWAV(
                noise_file,
                self.max_frames,
                evalmode=False,
            )
            noise_snr = random.uniform(*self.noise_snr[noise_category])
            noise_db = 10.0 * np.log10(np.mean(noise_audio[0] ** 2) + 1e-4)
            scale = np.sqrt(10.0 ** ((clean_db - noise_db - noise_snr) / 10.0))
            noises.append(scale * noise_audio)
        return np.sum(np.concatenate(noises, axis=0), axis=0, keepdims=True) + audio

    def reverberate(self, audio):
        """Свернуть сигнал со случайным импульсным откликом помещения."""
        if not self.rir_files:
            raise FileNotFoundError(ERROR_RIR_FILES_MISSING)
        rir_file = random.choice(self.rir_files)
        impulse_response, _ = sf.read(rir_file, always_2d=False)
        impulse_response = np.asarray(impulse_response, dtype=np.float32)
        if impulse_response.ndim == 2:
            impulse_response = impulse_response.mean(axis=1)
        impulse_response = np.expand_dims(impulse_response, 0)
        impulse_response /= np.sqrt(np.sum(impulse_response**2) + 1e-10)
        return signal.convolve(audio, impulse_response, mode="full")[
            :, : self.max_audio
        ]
