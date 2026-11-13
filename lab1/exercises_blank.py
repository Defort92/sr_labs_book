# Упражнения для лабораторной работы


# Импорт модулей
import numpy as np
from scipy.fftpack import dct

from constants import (
    ERROR_EMPHASIZED_SIGNAL_ONE_DIMENSIONAL,
    ERROR_FBANK_PARAMETERS_POSITIVE,
    ERROR_FILTER_BANK_FEATURES_TWO_DIMENSIONAL,
    ERROR_FRAME_PARAMETERS_POSITIVE,
    ERROR_FRAMES_TWO_DIMENSIONAL,
    ERROR_GENDER_CODE,
    ERROR_LINE_MUST_BE_STRING,
    ERROR_METADATA_LINE_FORMAT,
    ERROR_MFCC_COUNT,
    ERROR_NFFT_POSITIVE,
    ERROR_SIGNAL_ONE_DIMENSIONAL,
    ERROR_SPECTRUM_AND_FILTERS_TWO_DIMENSIONAL,
    ERROR_SPECTRUM_FILTER_DIMENSIONS,
    FRAME_SIZE_SECONDS,
    FRAME_STRIDE_SECONDS,
    NFFT,
    N_FILTERS,
    N_MFCC,
    PREEMPHASIS_COEFFICIENT,
    SUPPORTED_GENDER_CODES,
    TARGET_SAMPLE_RATE,
)


def split_meta_line(line, delimiter=' '):
    # Разбор строки формата «идентификатор_диктора пол путь»
    
    """
    :param line: строка метаданных
    :param delimiter: разделитель значений
    :return: идентификатор диктора, пол и путь к аудиофайлу
    """

    if not isinstance(line, str):
        raise TypeError(ERROR_LINE_MUST_BE_STRING)

    if delimiter == ' ':
        parts = line.strip().split(maxsplit=2)
    else:
        parts = line.strip().split(delimiter, maxsplit=2)

    if len(parts) != 3:
        raise ValueError(ERROR_METADATA_LINE_FORMAT)

    speaker_id, gender, file_path = parts

    if gender not in SUPPORTED_GENDER_CODES:
        raise ValueError(ERROR_GENDER_CODE)

    return speaker_id, gender, file_path

def preemphasis(signal, pre_emphasis=PREEMPHASIS_COEFFICIENT):
    # Выполнение преэмфазиса входного сигнала

    """
    :param signal: входной сигнал
    :param pre_emphasis: коэффициент преэмфазиса
    :return: сигнал после преэмфазиса
    """

    signal = np.asarray(signal)
    if signal.ndim != 1:
        raise ValueError(ERROR_SIGNAL_ONE_DIMENSIONAL)
    if signal.size == 0:
        return signal.astype(np.float64, copy=True)

    emphasized_signal = np.empty(signal.shape, dtype=np.float64)
    emphasized_signal[0] = signal[0]
    emphasized_signal[1:] = signal[1:] - pre_emphasis * signal[:-1]

    return emphasized_signal

def framing(
    emphasized_signal,
    sample_rate=TARGET_SAMPLE_RATE,
    frame_size=FRAME_SIZE_SECONDS,
    frame_stride=FRAME_STRIDE_SECONDS,
):
    # Разбиение сигнала на фреймы с применением окна Хэмминга
    
    """
    :param emphasized_signal: сигнал после преэмфазиса
    :param sample_rate: частота дискретизации
    :param frame_size: размер скользящего окна в секундах
    :param frame_stride: шаг скользящего окна в секундах
    :return: матрица фреймов [число фреймов x размер фрейма]
    """

    emphasized_signal = np.asarray(emphasized_signal)
    if emphasized_signal.ndim != 1:
        raise ValueError(ERROR_EMPHASIZED_SIGNAL_ONE_DIMENSIONAL)
    if sample_rate <= 0 or frame_size <= 0 or frame_stride <= 0:
        raise ValueError(ERROR_FRAME_PARAMETERS_POSITIVE)

    # Перевод значений из секунд в отсчёты сигнала
    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))

    if signal_length <= frame_length:
        num_frames = 1
    else:
        num_frames = 1 + int(np.ceil((signal_length - frame_length) / frame_step))

    pad_signal_length = (num_frames - 1) * frame_step + frame_length
    padding = np.zeros(pad_signal_length - signal_length)
    # Дополнение нулями сохраняет последний неполный фрейм.
    pad_signal = np.append(emphasized_signal, padding)

    frame_offsets = np.arange(num_frames)[:, np.newaxis] * frame_step
    sample_offsets = np.arange(frame_length)[np.newaxis, :]
    indices = frame_offsets + sample_offsets
    frames = pad_signal[indices]
    frames *= np.hamming(frame_length)

    return frames

def power_spectrum(frames, NFFT=NFFT):
    # Вычисление спектра мощности для каждого фрейма

    """
    :param frames: матрица фреймов сигнала
    :param NFFT: размер быстрого преобразования Фурье
    :return: спектр мощности для каждого фрейма
    """

    frames = np.asarray(frames)
    if frames.ndim != 2:
        raise ValueError(ERROR_FRAMES_TWO_DIMENSIONAL)
    if NFFT <= 0:
        raise ValueError(ERROR_NFFT_POSITIVE)

    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    pow_frames = mag_frames ** 2 / NFFT

    return pow_frames

def compute_fbank_filters(
    nfilt=N_FILTERS,
    sample_rate=TARGET_SAMPLE_RATE,
    NFFT=NFFT,
):
    # Построение треугольного Mel-банка фильтров

    """
    :param nfilt: количество фильтров
    :param sample_rate: частота дискретизации
    :param NFFT: размер быстрого преобразования Фурье
    :return: банк фильтров [число фильтров x (NFFT/2+1)]
    """
    
    low_freq_mel = 0
    high_freq = sample_rate / 2

    if nfilt <= 0 or sample_rate <= 0 or NFFT <= 0:
        raise ValueError(ERROR_FBANK_PARAMETERS_POSITIVE)

    high_freq_mel = 2595 * np.log10(1 + high_freq / 700)

    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)

    hz_points = 700 * (10 ** (mel_points / 2595) - 1)

    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])
        f_m = int(bin[m])
        f_m_plus = int(bin[m + 1])

        if f_m > f_m_minus:
            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        if f_m_plus > f_m:
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    return fbank

def compute_fbanks_features(pow_frames, fbank):
    # Вычисление логарифмов энергий на выходах Mel-фильтров
    
    """
    :param pow_frames: спектры мощности фреймов
    :param fbank: матрица Mel-фильтров
    :return: логарифмы энергий [число фреймов x число фильтров]
    """
    
    pow_frames = np.asarray(pow_frames)
    fbank = np.asarray(fbank)
    if pow_frames.ndim != 2 or fbank.ndim != 2:
        raise ValueError(ERROR_SPECTRUM_AND_FILTERS_TWO_DIMENSIONAL)
    if pow_frames.shape[1] != fbank.shape[1]:
        raise ValueError(ERROR_SPECTRUM_FILTER_DIMENSIONS)

    filter_banks_features = np.dot(pow_frames, fbank.T)

    filter_banks_features = np.where(filter_banks_features == 0, np.finfo(float).eps,
                                     filter_banks_features)  # Численная устойчивость
    filter_banks_features = np.log(filter_banks_features)

    return filter_banks_features

def compute_mfcc(filter_banks_features, num_ceps=N_MFCC):
    # Вычисление MFCC на основе заранее рассчитанных Log-Mel признаков
    
    """
    :param filter_banks_features: матрица Log-Mel признаков
    :param num_ceps: количество кепстральных коэффициентов
    :return: мел-частотные кепстральные коэффициенты
    """
    
    filter_banks_features = np.asarray(filter_banks_features)
    if filter_banks_features.ndim != 2:
        raise ValueError(ERROR_FILTER_BANK_FEATURES_TWO_DIMENSIONAL)
    if not 0 < num_ceps <= filter_banks_features.shape[1]:
        raise ValueError(ERROR_MFCC_COUNT)

    mfcc = dct(filter_banks_features, type=2, axis=1, norm='ortho')[:, :num_ceps]

    return mfcc

def mvn_floating(features, LC, RC, unbiased=False):
    # Локальная нормализация среднего и дисперсии признаков
    
    """
    :param features: матрица признаков [число фреймов x число признаков]
    :param LC: количество фреймов контекста слева
    :param RC: количество фреймов контекста справа
    :param unbiased: использовать несмещённую оценку дисперсии
    :return: нормализованная матрица признаков
    """
    
    nframes, dim = features.shape
    LC = min(LC, nframes - 1)
    RC = min(RC, nframes - 1)
    n = (np.r_[np.arange(RC + 1, nframes), np.ones(RC + 1) * nframes] - np.r_[np.zeros(LC), np.arange(nframes - LC)])[:,
        np.newaxis]
    f = np.cumsum(features, 0)
    s = np.cumsum(features ** 2, 0)
    f = (np.r_[f[RC:], np.repeat(f[[-1]], RC, axis=0)] - np.r_[np.zeros((LC + 1, dim)), f[:-LC - 1]]) / n
    s = (np.r_[s[RC:], np.repeat(s[[-1]], RC, axis=0)] - np.r_[np.zeros((LC + 1, dim)), s[:-LC - 1]]
         ) / (n - 1 if unbiased else n) - f ** 2 * (n / (n - 1) if unbiased else 1)
    
    variance = np.maximum(s, 0)
    std = np.sqrt(variance)
    normalised_features = np.zeros_like(features, dtype=np.float64)
    np.divide(
        features - f,
        std,
        out=normalised_features,
        where=std > np.finfo(float).eps,
    )

    return normalised_features
