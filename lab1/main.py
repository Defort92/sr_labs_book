"""Запуск первой лабораторной работы и сохранение полученных графиков."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from constants import (
    ANALYSIS_DURATION_SECONDS,
    CLI_DEMO_HELP,
    CLI_DIRECTORY_METAVAR,
    CLI_DESCRIPTION,
    CLI_HELP_HELP,
    CLI_METADATA_HELP,
    CLI_OPTIONS_PREFIX,
    CLI_OPTIONS_TITLE,
    CLI_OUTPUT_DIR_HELP,
    CLI_PATH_METAVAR,
    CLI_RUSSIAN_OPTIONS_PREFIX,
    CLI_RUSSIAN_USAGE_PREFIX,
    CLI_USAGE_PREFIX,
    DEFAULT_METADATA_PATH,
    DEFAULT_RESULTS_DIR,
    DEMO_COMPLETE_MESSAGE,
    DEMO_ENVELOPE_FREQUENCY,
    DEMO_FEMALE_BASE_FREQUENCY,
    DEMO_FREQUENCY_STEP,
    DEMO_HARMONICS,
    DEMO_MALE_BASE_FREQUENCY,
    DEMO_NOISE_LEVEL,
    DEMO_RANDOM_SEED,
    DEMO_RECORDS_PER_GENDER,
    ERROR_AUDIO_DIMENSION,
    FEATURE_MAPS_PLOT_FILENAME,
    FEATURE_NAME_LOG_MEL,
    FEATURE_NAME_MFCC,
    FEMALE_GENDER_CODE,
    FRAME_SIZE_MILLISECONDS,
    FRAME_STRIDE_MILLISECONDS,
    GENDER_LOG_MEL_PLOT_FILENAME,
    GENDER_MFCC_PLOT_FILENAME,
    HISTOGRAM_MAX_BINS,
    HISTOGRAM_MIN_BINS,
    LABEL_AMPLITUDE,
    LABEL_COMPONENT,
    LABEL_DENSITY,
    LABEL_FEATURE_VALUE,
    LABEL_FEMALE,
    LABEL_FILTER_WEIGHT,
    LABEL_FRAME,
    LABEL_FREQUENCY_HZ,
    LABEL_MALE,
    LABEL_TIME_SECONDS,
    LAB_COMPLETE_MESSAGE,
    LAB_DIR,
    MALE_GENDER_CODE,
    MEL_FILTER_BANK_PLOT_FILENAME,
    MISSING_AUDIO_MESSAGE,
    MODE_SYNTHETIC_DEMO,
    MODE_VOXCELEB,
    MVN_LEFT_CONTEXT,
    MVN_RIGHT_CONTEXT,
    NFFT,
    N_FILTERS,
    N_MFCC,
    PLOT_DPI,
    PREEMPHASIS_PLOT_FILENAME,
    SPECTROGRAM_NFFT,
    SPECTROGRAM_OVERLAP,
    SUMMARY_FILENAME,
    SUMMARY_KEY_FEMALE_RECORDINGS,
    SUMMARY_KEY_FRAME_MS,
    SUMMARY_KEY_FRAME_SHIFT_MS,
    SUMMARY_KEY_LOG_MEL_SHAPE,
    SUMMARY_KEY_MALE_RECORDINGS,
    SUMMARY_KEY_MEL_FILTERS,
    SUMMARY_KEY_MFCC,
    SUMMARY_KEY_MFCC_SHAPE,
    SUMMARY_KEY_MODE,
    SUMMARY_KEY_NFFT,
    SUMMARY_KEY_PARAMETERS,
    SUMMARY_KEY_RECORDINGS,
    SUMMARY_KEY_SAMPLE_RATE,
    TARGET_SAMPLE_RATE,
    TITLE_AFTER_PREEMPHASIS,
    TITLE_GENDER_DISTRIBUTIONS,
    TITLE_LOG_MEL,
    TITLE_MEL_FILTER_BANK,
    TITLE_MFCC,
    TITLE_NORMALIZED_LOG_MEL,
    TITLE_NORMALIZED_MFCC,
    TITLE_ORIGINAL_SIGNAL,
    TITLE_ORIGINAL_WAVEFORM,
)

from exercises_blank import (
    compute_fbank_filters,
    compute_fbanks_features,
    compute_mfcc,
    framing,
    mvn_floating,
    power_spectrum,
    preemphasis,
    split_meta_line,
)


def configure_console_encoding() -> None:
    """Настроить корректный вывод русского текста в консоль Windows."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


class RussianArgumentParser(argparse.ArgumentParser):
    """Парсер аргументов с русскими заголовками справки."""

    def format_usage(self) -> str:
        return super().format_usage().replace(
            CLI_USAGE_PREFIX,
            CLI_RUSSIAN_USAGE_PREFIX,
            1,
        )

    def format_help(self) -> str:
        return (
            super()
            .format_help()
            .replace(CLI_USAGE_PREFIX, CLI_RUSSIAN_USAGE_PREFIX, 1)
            .replace(CLI_OPTIONS_PREFIX, CLI_RUSSIAN_OPTIONS_PREFIX, 1)
        )

def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """Преобразовать сигнал в моно и нормализовать его амплитуду."""
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim == 2:
        signal = signal.mean(axis=1)
    if signal.ndim != 1:
        raise ValueError(ERROR_AUDIO_DIMENSION)

    peak = np.max(np.abs(signal)) if signal.size else 0.0
    return signal / peak if peak > 0 else signal


def load_audio(path: Path) -> np.ndarray:
    """Загрузить аудио, преобразовать в моно и привести к частоте 16 кГц."""
    signal, sample_rate = sf.read(path, always_2d=False)
    signal = normalize_signal(signal)
    if sample_rate != TARGET_SAMPLE_RATE:
        divisor = np.gcd(sample_rate, TARGET_SAMPLE_RATE)
        signal = resample_poly(
            signal,
            TARGET_SAMPLE_RATE // divisor,
            sample_rate // divisor,
        )
    return normalize_signal(signal)


def extract_features(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Вычислить Log-Mel признаки и MFCC для одного сигнала."""
    emphasized = preemphasis(signal)
    frames = framing(emphasized, sample_rate=TARGET_SAMPLE_RATE)
    spectrum = power_spectrum(frames, NFFT=NFFT)
    filters = compute_fbank_filters(
        nfilt=N_FILTERS,
        sample_rate=TARGET_SAMPLE_RATE,
        NFFT=NFFT,
    )
    filter_banks = compute_fbanks_features(spectrum, filters)
    mfcc = compute_mfcc(filter_banks, num_ceps=N_MFCC)
    return filter_banks, mfcc


def read_metadata(metadata_path: Path) -> list[tuple[str, str, Path]]:
    """Прочитать метаданные и получить абсолютные пути к аудиофайлам."""
    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        lines = metadata_file.readlines()

    records = []
    for line in lines[1:]:
        if not line.strip():
            continue
        speaker_id, gender, file_path = split_meta_line(line)
        audio_path = Path(file_path)
        if not audio_path.is_absolute():
            audio_path = (LAB_DIR / audio_path).resolve()
        records.append((speaker_id, gender, audio_path))
    return records


def generate_demo_records() -> list[tuple[str, str, np.ndarray]]:
    """Создать речеподобные сигналы для демонстрационной проверки."""
    rng = np.random.default_rng(DEMO_RANDOM_SEED)
    duration = ANALYSIS_DURATION_SECONDS
    time = np.arange(int(duration * TARGET_SAMPLE_RATE)) / TARGET_SAMPLE_RATE
    records = []

    gender_frequencies = (
        (MALE_GENDER_CODE, DEMO_MALE_BASE_FREQUENCY),
        (FEMALE_GENDER_CODE, DEMO_FEMALE_BASE_FREQUENCY),
    )
    for gender, base_frequency in gender_frequencies:
        for index in range(DEMO_RECORDS_PER_GENDER):
            fundamental = base_frequency + index * DEMO_FREQUENCY_STEP
            envelope = (
                0.55
                + 0.45
                * np.sin(2 * np.pi * DEMO_ENVELOPE_FREQUENCY * time) ** 2
            )
            signal = sum(
                np.sin(2 * np.pi * fundamental * harmonic * time) / harmonic
                for harmonic in range(1, DEMO_HARMONICS + 1)
            )
            signal = (
                envelope * signal
                + DEMO_NOISE_LEVEL * rng.standard_normal(time.size)
            )
            records.append(
                (f"демо_{gender}_{index}", gender, normalize_signal(signal))
            )
    return records


def plot_signal_comparison(signal: np.ndarray, output_path: Path) -> None:
    """Сохранить осциллограмму и спектрограммы до и после преэмфазиса."""
    emphasized = preemphasis(signal)
    time = np.arange(signal.size) / TARGET_SAMPLE_RATE

    figure, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)
    axes[0].plot(time, signal, linewidth=0.6)
    axes[0].set(
        title=TITLE_ORIGINAL_WAVEFORM,
        xlabel=LABEL_TIME_SECONDS,
        ylabel=LABEL_AMPLITUDE,
    )
    axes[0].grid(True)
    axes[1].specgram(
        signal,
        NFFT=SPECTROGRAM_NFFT,
        Fs=TARGET_SAMPLE_RATE,
        noverlap=SPECTROGRAM_OVERLAP,
    )
    axes[1].set(
        title=TITLE_ORIGINAL_SIGNAL,
        xlabel=LABEL_TIME_SECONDS,
        ylabel=LABEL_FREQUENCY_HZ,
    )
    axes[2].specgram(
        emphasized,
        NFFT=SPECTROGRAM_NFFT,
        Fs=TARGET_SAMPLE_RATE,
        noverlap=SPECTROGRAM_OVERLAP,
    )
    axes[2].set(
        title=TITLE_AFTER_PREEMPHASIS,
        xlabel=LABEL_TIME_SECONDS,
        ylabel=LABEL_FREQUENCY_HZ,
    )
    figure.savefig(output_path, dpi=PLOT_DPI)
    plt.close(figure)


def plot_filter_bank(output_path: Path) -> None:
    """Сохранить частотные характеристики всех Mel-фильтров."""
    filters = compute_fbank_filters(N_FILTERS, TARGET_SAMPLE_RATE, NFFT)
    frequencies = np.linspace(0, TARGET_SAMPLE_RATE / 2, filters.shape[1])
    figure, axis = plt.subplots(figsize=(12, 5), constrained_layout=True)
    axis.plot(frequencies, filters.T)
    axis.set(
        title=TITLE_MEL_FILTER_BANK,
        xlabel=LABEL_FREQUENCY_HZ,
        ylabel=LABEL_FILTER_WEIGHT,
    )
    axis.grid(True)
    figure.savefig(output_path, dpi=PLOT_DPI)
    plt.close(figure)


def plot_feature_maps(
    filter_banks: np.ndarray,
    mfcc: np.ndarray,
    output_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Сохранить исходные и нормализованные карты признаков."""
    filter_banks_mvn = mvn_floating(
        filter_banks,
        MVN_LEFT_CONTEXT,
        MVN_RIGHT_CONTEXT,
    )
    mfcc_mvn = mvn_floating(mfcc, MVN_LEFT_CONTEXT, MVN_RIGHT_CONTEXT)

    figure, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    images = (
        axes[0, 0].imshow(filter_banks.T, origin="lower", aspect="auto"),
        axes[0, 1].imshow(mfcc.T, origin="lower", aspect="auto"),
        axes[1, 0].imshow(filter_banks_mvn.T, origin="lower", aspect="auto"),
        axes[1, 1].imshow(mfcc_mvn.T, origin="lower", aspect="auto"),
    )
    titles = (
        TITLE_LOG_MEL,
        TITLE_MFCC,
        TITLE_NORMALIZED_LOG_MEL,
        TITLE_NORMALIZED_MFCC,
    )
    for axis, image, title_text in zip(axes.flat, images, titles):
        axis.set(
            title=title_text,
            xlabel=LABEL_FRAME,
            ylabel=LABEL_COMPONENT,
        )
        figure.colorbar(image, ax=axis, fraction=0.025)
    figure.savefig(output_path, dpi=PLOT_DPI)
    plt.close(figure)
    return filter_banks_mvn, mfcc_mvn


def plot_gender_distributions(
    male: np.ndarray,
    female: np.ndarray,
    feature_name: str,
    output_path: Path,
) -> None:
    """Построить первые три компоненты признаков для мужских и женских голосов."""
    figure, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
    for component, axis in enumerate(axes):
        lower = min(male[:, component].min(), female[:, component].min())
        upper = max(male[:, component].max(), female[:, component].max())
        bins = min(
            HISTOGRAM_MAX_BINS,
            max(
                HISTOGRAM_MIN_BINS,
                int(np.sqrt(male.shape[0] + female.shape[0])),
            ),
        )
        axis.hist(
            male[:, component],
            bins=bins,
            range=(lower, upper),
            density=True,
            histtype="step",
            color="tab:blue",
            label=LABEL_MALE,
        )
        axis.hist(
            female[:, component],
            bins=bins,
            range=(lower, upper),
            density=True,
            histtype="step",
            color="tab:red",
            label=LABEL_FEMALE,
        )
        axis.set(
            title=(
                f"{TITLE_GENDER_DISTRIBUTIONS}: "
                f"{feature_name}, компонента {component + 1}"
            ),
            xlabel=LABEL_FEATURE_VALUE,
            ylabel=LABEL_DENSITY,
        )
        axis.grid(True)
        axis.legend()
    figure.savefig(output_path, dpi=PLOT_DPI)
    plt.close(figure)


def run(records: list[tuple[str, str, np.ndarray]], output_dir: Path, mode: str) -> None:
    """Выполнить эксперименты и сохранить графики со сводкой."""
    output_dir.mkdir(parents=True, exist_ok=True)
    first_signal = records[0][2][
        : int(ANALYSIS_DURATION_SECONDS * TARGET_SAMPLE_RATE)
    ]
    plot_signal_comparison(first_signal, output_dir / PREEMPHASIS_PLOT_FILENAME)
    plot_filter_bank(output_dir / MEL_FILTER_BANK_PLOT_FILENAME)

    first_filter_banks, first_mfcc = extract_features(first_signal)
    plot_feature_maps(
        first_filter_banks,
        first_mfcc,
        output_dir / FEATURE_MAPS_PLOT_FILENAME,
    )

    male_filter_bank_parts = []
    female_filter_bank_parts = []
    male_mfcc_parts = []
    female_mfcc_parts = []
    for _, gender, signal in records:
        filter_banks, mfcc = extract_features(signal)
        if gender == MALE_GENDER_CODE:
            male_filter_bank_parts.append(filter_banks)
            male_mfcc_parts.append(mfcc)
        else:
            female_filter_bank_parts.append(filter_banks)
            female_mfcc_parts.append(mfcc)

    male_filter_banks = np.concatenate(male_filter_bank_parts)
    female_filter_banks = np.concatenate(female_filter_bank_parts)
    male_mfcc = np.concatenate(male_mfcc_parts)
    female_mfcc = np.concatenate(female_mfcc_parts)

    plot_gender_distributions(
        male_filter_banks,
        female_filter_banks,
        FEATURE_NAME_LOG_MEL,
        output_dir / GENDER_LOG_MEL_PLOT_FILENAME,
    )
    plot_gender_distributions(
        male_mfcc,
        female_mfcc,
        FEATURE_NAME_MFCC,
        output_dir / GENDER_MFCC_PLOT_FILENAME,
    )

    summary = {
        SUMMARY_KEY_MODE: mode,
        SUMMARY_KEY_RECORDINGS: len(records),
        SUMMARY_KEY_MALE_RECORDINGS: sum(
            gender == MALE_GENDER_CODE for _, gender, _ in records
        ),
        SUMMARY_KEY_FEMALE_RECORDINGS: sum(
            gender == FEMALE_GENDER_CODE for _, gender, _ in records
        ),
        SUMMARY_KEY_LOG_MEL_SHAPE: list(first_filter_banks.shape),
        SUMMARY_KEY_MFCC_SHAPE: list(first_mfcc.shape),
        SUMMARY_KEY_PARAMETERS: {
            SUMMARY_KEY_SAMPLE_RATE: TARGET_SAMPLE_RATE,
            SUMMARY_KEY_FRAME_MS: FRAME_SIZE_MILLISECONDS,
            SUMMARY_KEY_FRAME_SHIFT_MS: FRAME_STRIDE_MILLISECONDS,
            SUMMARY_KEY_NFFT: NFFT,
            SUMMARY_KEY_MEL_FILTERS: N_FILTERS,
            SUMMARY_KEY_MFCC: N_MFCC,
        },
    }
    with (output_dir / SUMMARY_FILENAME).open(
        "w",
        encoding="utf-8",
    ) as summary_file:
        json.dump(summary, summary_file, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = RussianArgumentParser(
        description=CLI_DESCRIPTION,
        add_help=False,
    )
    parser._optionals.title = CLI_OPTIONS_TITLE
    parser.add_argument(
        "-h",
        "--help",
        action="help",
        help=CLI_HELP_HELP,
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        metavar=CLI_PATH_METAVAR,
        help=CLI_METADATA_HELP,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        metavar=CLI_DIRECTORY_METAVAR,
        help=CLI_OUTPUT_DIR_HELP,
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help=CLI_DEMO_HELP,
    )
    return parser.parse_args()


def main() -> None:
    configure_console_encoding()
    args = parse_args()
    if args.demo:
        run(generate_demo_records(), args.output_dir, mode=MODE_SYNTHETIC_DEMO)
        print(DEMO_COMPLETE_MESSAGE.format(path=args.output_dir.resolve()))
        return

    metadata_records = read_metadata(args.metadata.resolve())
    missing = [path for _, _, path in metadata_records if not path.is_file()]
    if missing:
        examples = "\n".join(f"  - {path}" for path in missing[:3])
        print(
            MISSING_AUDIO_MESSAGE.format(examples=examples),
            file=sys.stderr,
        )
        raise SystemExit(1)

    records = [
        (speaker_id, gender, load_audio(path))
        for speaker_id, gender, path in metadata_records
    ]
    run(records, args.output_dir, mode=MODE_VOXCELEB)
    print(LAB_COMPLETE_MESSAGE.format(path=args.output_dir.resolve()))


if __name__ == "__main__":
    main()
