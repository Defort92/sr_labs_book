"""Упражнения для второй лабораторной работы."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.signal
from scipy.ndimage import maximum_filter1d, minimum_filter1d

from constants import (
    ERROR_ENERGY_EMPTY,
    ERROR_FRAMES_TWO_DIMENSIONAL,
    ERROR_GMM_DIMENSIONS,
    ERROR_GMM_REALIGNMENTS_POSITIVE,
    ERROR_IMPULSE_RESPONSE_EMPTY,
    ERROR_NOISE_SIGMA,
    ERROR_RTTM_FORMAT,
    ERROR_SIGNAL_EMPTY,
    ERROR_SIGNAL_ONE_DIMENSIONAL,
    ERROR_WINDOW_AND_SHIFT_POSITIVE,
    FRAME_SHIFT,
    FRAME_WINDOW,
    GMM_INITIAL_MEANS,
    GMM_INITIAL_SIGMAS,
    GMM_INITIAL_WEIGHTS,
    GMM_REALIGNMENTS,
    MORPHOLOGY_MASK_SIZE,
    NUMERICAL_EPSILON,
    RTTM_RECORD_TYPE,
    VAD_THRESHOLD,
)


def _as_one_dimensional(signal, error_message):
    """Преобразовать вход в одномерный массив и проверить его форму."""
    array = np.asarray(signal)
    if array.ndim != 1:
        raise ValueError(error_message)
    if array.size == 0:
        raise ValueError(ERROR_SIGNAL_EMPTY)
    return array


def load_vad_markup(path_to_rttm, signal, fs):
    """Прочитать RTTM и сформировать посэмпловую бинарную VAD-разметку."""
    signal = _as_one_dimensional(signal, ERROR_SIGNAL_ONE_DIMENSIONAL)
    vad_markup = np.zeros(signal.size, dtype=np.float32)

    with Path(path_to_rttm).open("r", encoding="utf-8") as rttm_file:
        for raw_line in rttm_file:
            line = raw_line.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) < 5 or fields[0] != RTTM_RECORD_TYPE:
                raise ValueError(ERROR_RTTM_FORMAT.format(line=line))

            try:
                start_seconds = float(fields[3])
                duration_seconds = float(fields[4])
            except ValueError as error:
                raise ValueError(ERROR_RTTM_FORMAT.format(line=line)) from error

            start_sample = max(0, int(round(start_seconds * fs)))
            end_sample = min(
                signal.size,
                int(round((start_seconds + duration_seconds) * fs)),
            )
            if end_sample > start_sample:
                vad_markup[start_sample:end_sample] = 1.0

    return vad_markup


def framing(signal, window=FRAME_WINDOW, shift=FRAME_SHIFT):
    """Разбить одномерный сигнал на перекрывающиеся фреймы."""
    signal = _as_one_dimensional(signal, ERROR_SIGNAL_ONE_DIMENSIONAL)
    if window <= 0 or shift <= 0:
        raise ValueError(ERROR_WINDOW_AND_SHIFT_POSITIVE)

    if signal.size <= window:
        frame_count = 1
    else:
        frame_count = 1 + int(np.ceil((signal.size - window) / shift))

    padded_length = (frame_count - 1) * shift + window
    padded_signal = np.pad(signal, (0, padded_length - signal.size))
    offsets = np.arange(frame_count)[:, np.newaxis] * shift
    indices = offsets + np.arange(window)[np.newaxis, :]
    return padded_signal[indices].astype(np.float32, copy=False)


def frame_energy(frames):
    """Вычислить суммарную энергию каждого фрейма."""
    frames = np.asarray(frames)
    if frames.ndim != 2:
        raise ValueError(ERROR_FRAMES_TWO_DIMENSIONAL)
    if frames.shape[0] == 0:
        raise ValueError(ERROR_ENERGY_EMPTY)
    return np.sum(frames, axis=1, dtype=np.float64).astype(np.float32)


def norm_energy(energy):
    """Нормализовать энергию с помощью среднего и стандартного отклонения."""
    energy = np.asarray(energy, dtype=np.float64)
    if energy.ndim != 1:
        raise ValueError(ERROR_SIGNAL_ONE_DIMENSIONAL)
    if energy.size == 0:
        raise ValueError(ERROR_ENERGY_EMPTY)

    standard_deviation = energy.std()
    if standard_deviation <= NUMERICAL_EPSILON:
        return np.zeros(energy.shape, dtype=np.float32)
    return ((energy - energy.mean()) / standard_deviation).astype(np.float32)


def gmm_train(energy, gauss_pdf, n_realignment=GMM_REALIGNMENTS):
    """Обучить параметры трёхкомпонентной GMM с помощью EM-алгоритма."""
    energy = np.asarray(energy, dtype=np.float64)
    if energy.ndim != 1:
        raise ValueError(ERROR_SIGNAL_ONE_DIMENSIONAL)
    if energy.size == 0:
        raise ValueError(ERROR_ENERGY_EMPTY)
    if n_realignment <= 0:
        raise ValueError(ERROR_GMM_REALIGNMENTS_POSITIVE)

    weights = np.asarray(GMM_INITIAL_WEIGHTS, dtype=np.float64).copy()
    means = np.asarray(GMM_INITIAL_MEANS, dtype=np.float64).copy()
    sigmas = np.asarray(GMM_INITIAL_SIGMAS, dtype=np.float64).copy()
    if not (weights.size == means.size == sigmas.size):
        raise ValueError(ERROR_GMM_DIMENSIONS)

    responsibilities = np.zeros((energy.size, weights.size), dtype=np.float64)
    for _ in range(n_realignment):
        # E-шаг: вероятность принадлежности фрейма каждому компоненту.
        for component in range(weights.size):
            responsibilities[:, component] = weights[component] * gauss_pdf(
                energy,
                means[component],
                sigmas[component],
            )
        denominator = responsibilities.sum(axis=1, keepdims=True)
        denominator = np.maximum(denominator, NUMERICAL_EPSILON)
        responsibilities /= denominator

        # M-шаг: пересчёт весов, средних и стандартных отклонений.
        component_mass = responsibilities.sum(axis=0)
        component_mass = np.maximum(component_mass, NUMERICAL_EPSILON)
        weights = component_mass / energy.size
        means = (responsibilities * energy[:, np.newaxis]).sum(axis=0)
        means /= component_mass
        differences = energy[:, np.newaxis] - means[np.newaxis, :]
        variances = (responsibilities * differences ** 2).sum(axis=0)
        variances /= component_mass
        sigmas = np.sqrt(np.maximum(variances, NUMERICAL_EPSILON))

    # Первый компонент должен соответствовать наименьшей энергии, то есть паузе.
    order = np.argsort(means)
    return weights[order], means[order], sigmas[order]


def eval_frame_post_prob(energy, gauss_pdf, weights, means, sigmas):
    """Оценить апостериорную вероятность отсутствия речи во фрейме."""
    energy = np.asarray(energy, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    sigmas = np.asarray(sigmas, dtype=np.float64)
    if not (weights.size == means.size == sigmas.size):
        raise ValueError(ERROR_GMM_DIMENSIONS)

    weighted_probabilities = np.column_stack(
        [
            weights[component]
            * gauss_pdf(energy, means[component], sigmas[component])
            for component in range(weights.size)
        ]
    )
    denominator = np.maximum(
        weighted_probabilities.sum(axis=1),
        NUMERICAL_EPSILON,
    )
    silence_component = int(np.argmin(means))
    return weighted_probabilities[:, silence_component] / denominator


def frames_to_samples(frame_markup, signal_length, shift):
    """Преобразовать пофреймовую разметку в посэмпловую."""
    sample_markup = np.zeros(signal_length, dtype=np.float32)
    for index, value in enumerate(frame_markup):
        start = index * shift
        if start >= signal_length:
            break
        end = min(start + shift, signal_length)
        sample_markup[start:end] = value
    covered = min(frame_markup.size * shift, signal_length)
    if covered < signal_length:
        sample_markup[covered:] = frame_markup[-1]
    return sample_markup


def morphological_cleanup(sample_markup, mask_size=MORPHOLOGY_MASK_SIZE):
    """Выполнить одномерные закрытие и открытие за линейное время."""
    binary_markup = sample_markup.astype(np.uint8)
    # Закрытие заполняет короткие паузы внутри речевых участков.
    dilated = maximum_filter1d(
        binary_markup,
        size=mask_size,
        mode="constant",
        cval=0,
    )
    closed = minimum_filter1d(
        dilated,
        size=mask_size,
        mode="constant",
        cval=0,
    )
    # Открытие удаляет короткие одиночные срабатывания детектора.
    eroded = minimum_filter1d(
        closed,
        size=mask_size,
        mode="constant",
        cval=0,
    )
    opened = maximum_filter1d(
        eroded,
        size=mask_size,
        mode="constant",
        cval=0,
    )
    return opened.astype(np.float32)


def energy_gmm_vad(
    signal,
    window=FRAME_WINDOW,
    shift=FRAME_SHIFT,
    gauss_pdf=None,
    n_realignment=GMM_REALIGNMENTS,
    vad_thr=VAD_THRESHOLD,
    mask_size_morph_filt=MORPHOLOGY_MASK_SIZE,
):
    """Построить VAD-разметку энергетическим детектором на основе GMM."""
    signal = _as_one_dimensional(signal, ERROR_SIGNAL_ONE_DIMENSIONAL)
    if gauss_pdf is None:
        gauss_pdf = gaussian_pdf

    squared_signal = signal.astype(np.float64) ** 2
    frames = framing(squared_signal, window=window, shift=shift)
    energy = frame_energy(frames)
    normalized_energy = norm_energy(energy)
    weights, means, sigmas = gmm_train(
        normalized_energy,
        gauss_pdf,
        n_realignment=n_realignment,
    )
    silence_probability = eval_frame_post_prob(
        normalized_energy,
        gauss_pdf,
        weights,
        means,
        sigmas,
    )
    frame_markup = (silence_probability < vad_thr).astype(np.float32)
    sample_markup = frames_to_samples(frame_markup, signal.size, shift)

    if mask_size_morph_filt > 1:
        sample_markup = morphological_cleanup(
            sample_markup,
            int(mask_size_morph_filt),
        )

    return sample_markup.astype(np.float32)


def gaussian_pdf(value, mean, sigma):
    """Вычислить плотность нормального распределения."""
    safe_sigma = max(abs(float(sigma)), NUMERICAL_EPSILON)
    coefficient = 1.0 / (safe_sigma * np.sqrt(2.0 * np.pi))
    return coefficient * np.exp(-((value - mean) ** 2) / (2.0 * safe_sigma ** 2))


def reverb(signal, impulse_response):
    """Смоделировать реверберацию свёрткой с импульсным откликом."""
    signal = _as_one_dimensional(signal, ERROR_SIGNAL_ONE_DIMENSIONAL)
    impulse_response = np.asarray(impulse_response, dtype=np.float64)
    if impulse_response.ndim != 1 or impulse_response.size == 0:
        raise ValueError(ERROR_IMPULSE_RESPONSE_EMPTY)

    impulse_norm = np.linalg.norm(impulse_response)
    if impulse_norm > NUMERICAL_EPSILON:
        impulse_response = impulse_response / impulse_norm
    reverberated = scipy.signal.fftconvolve(signal, impulse_response, mode="full")
    reverberated = reverberated[: signal.size]

    original_peak = np.max(np.abs(signal))
    result_peak = np.max(np.abs(reverberated))
    if result_peak > NUMERICAL_EPSILON:
        reverberated *= original_peak / result_peak
    return reverberated.astype(np.float32)


def awgn(signal, sigma_noise, rng=None):
    """Добавить к сигналу белый гауссовский шум."""
    signal = _as_one_dimensional(signal, ERROR_SIGNAL_ONE_DIMENSIONAL)
    if sigma_noise < 0:
        raise ValueError(ERROR_NOISE_SIGMA)
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0.0, sigma_noise, signal.size)
    return (signal + noise).astype(np.float32)
