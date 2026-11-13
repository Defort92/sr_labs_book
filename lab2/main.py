"""Запуск второй лабораторной работы и сохранение результатов."""

from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from constants import (
    AUGMENTATION_AWGN_TEMPLATE,
    AUGMENTATION_CLEAN,
    AUGMENTATION_REVERB,
    AWGN_LEVELS,
    CLI_DEMO_HELP,
    CLI_DESCRIPTION,
    CLI_DIRECTORY_METAVAR,
    CLI_HELP_HELP,
    CLI_OPTIONS_PREFIX,
    CLI_OPTIONS_TITLE,
    CLI_OUTPUT_DIR_HELP,
    CLI_RTTM_DIR_HELP,
    CLI_RUSSIAN_OPTIONS_PREFIX,
    CLI_RUSSIAN_USAGE_PREFIX,
    CLI_USAGE_PREFIX,
    CLI_VOXCELEB_DIR_HELP,
    DEFAULT_RESULTS_DIR,
    DEFAULT_RTTM_DIR,
    DEFAULT_VOXCELEB_DIR,
    DEMO_COMPLETE_MESSAGE,
    DEMO_DURATION_SECONDS,
    DEMO_RANDOM_SEED,
    DEMO_SILENCE_NOISE_LEVEL,
    DEMO_SPEECH_FREQUENCIES,
    DEMO_SPEECH_INTERVALS,
    DEMO_SPEECH_NOISE_LEVEL,
    ENERGY_PLOT_FILENAME,
    ERROR_AUDIO_MISSING,
    ERROR_MARKUP_LENGTHS,
    ERROR_NO_RTTM_FILES,
    FRAME_SHIFT,
    FRAME_WINDOW,
    GMM_COMPONENT_LABEL_TEMPLATE,
    GMM_REALIGNMENTS,
    IMPULSE_RESPONSE_PATH,
    LABEL_AMPLITUDE,
    LABEL_DENSITY,
    LABEL_FILTERED_VAD,
    LABEL_FNR,
    LABEL_FPR,
    LABEL_IDEAL_VAD,
    LABEL_METRIC_VALUE,
    LABEL_NORMALIZED_ENERGY,
    LABEL_PRECISION,
    LABEL_RAW_VAD,
    LABEL_RECALL,
    LABEL_SIGNAL,
    LABEL_TIME_SECONDS,
    LAB_COMPLETE_MESSAGE,
    METRICS_MESSAGE,
    METRICS_PLOT_FILENAME,
    MODE_SYNTHETIC_DEMO,
    MODE_VOXCELEB,
    MORPHOLOGY_MASK_SIZE,
    NUMERICAL_EPSILON,
    PLOT_DPI,
    SUMMARY_FILENAME,
    SUMMARY_KEY_FN,
    SUMMARY_KEY_FNR,
    SUMMARY_KEY_FP,
    SUMMARY_KEY_FPR,
    SUMMARY_KEY_MORPHOLOGY,
    SUMMARY_KEY_PARAMETERS,
    SUMMARY_KEY_PRECISION,
    SUMMARY_KEY_REALIGNMENTS,
    SUMMARY_KEY_RECALL,
    SUMMARY_KEY_RESULTS,
    SUMMARY_KEY_SHIFT,
    SUMMARY_KEY_THRESHOLD,
    SUMMARY_KEY_TN,
    SUMMARY_KEY_TP,
    SUMMARY_KEY_WINDOW,
    SUMMARY_KEY_MODE,
    TARGET_SAMPLE_RATE,
    TITLE_DISTORTION_METRICS,
    TITLE_ENERGY_HISTOGRAM,
    TITLE_VAD_MARKUP,
    VAD_PLOT_FILENAME,
    VAD_THRESHOLD,
)
from exercises_blank import (
    awgn,
    energy_gmm_vad,
    eval_frame_post_prob,
    frame_energy,
    framing,
    gaussian_pdf,
    gmm_train,
    load_vad_markup,
    norm_energy,
    reverb,
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
    """Преобразовать аудио в моно и нормализовать пиковую амплитуду."""
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim == 2:
        signal = signal.mean(axis=1)
    peak = np.max(np.abs(signal)) if signal.size else 0.0
    return signal / peak if peak > NUMERICAL_EPSILON else signal


def load_audio(path: Path, target_sample_rate=TARGET_SAMPLE_RATE) -> np.ndarray:
    """Загрузить аудиофайл и привести его к требуемой частоте дискретизации."""
    signal, sample_rate = sf.read(path, always_2d=False)
    signal = normalize_signal(signal)
    if sample_rate != target_sample_rate:
        divisor = np.gcd(sample_rate, target_sample_rate)
        signal = resample_poly(
            signal,
            target_sample_rate // divisor,
            sample_rate // divisor,
        )
    return normalize_signal(signal)


def calculate_metrics(reference: np.ndarray, hypothesis: np.ndarray) -> dict:
    """Вычислить матрицу ошибок, FNR, FPR, точность и полноту."""
    reference = np.asarray(reference, dtype=bool)
    hypothesis = np.asarray(hypothesis, dtype=bool)
    if reference.shape != hypothesis.shape:
        raise ValueError(ERROR_MARKUP_LENGTHS)

    true_positive = int(np.sum(reference & hypothesis))
    false_positive = int(np.sum(~reference & hypothesis))
    false_negative = int(np.sum(reference & ~hypothesis))
    true_negative = int(np.sum(~reference & ~hypothesis))
    return metrics_from_counts(
        true_positive,
        false_positive,
        false_negative,
        true_negative,
    )


def metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> dict:
    """Преобразовать элементы матрицы ошибок в итоговые метрики."""
    def safe_divide(numerator, denominator):
        return float(numerator / denominator) if denominator else 0.0

    return {
        SUMMARY_KEY_TP: int(tp),
        SUMMARY_KEY_FP: int(fp),
        SUMMARY_KEY_FN: int(fn),
        SUMMARY_KEY_TN: int(tn),
        SUMMARY_KEY_FNR: safe_divide(fn, fn + tp),
        SUMMARY_KEY_FPR: safe_divide(fp, fp + tn),
        SUMMARY_KEY_PRECISION: safe_divide(tp, tp + fp),
        SUMMARY_KEY_RECALL: safe_divide(tp, tp + fn),
    }


def add_metric_counts(target: dict, source: dict) -> None:
    """Добавить элементы одной матрицы ошибок к другой."""
    for key in (
        SUMMARY_KEY_TP,
        SUMMARY_KEY_FP,
        SUMMARY_KEY_FN,
        SUMMARY_KEY_TN,
    ):
        target[key] = target.get(key, 0) + source[key]


def finalize_metric_counts(counts: dict) -> dict:
    """Вычислить метрики для накопленной матрицы ошибок."""
    return metrics_from_counts(
        counts[SUMMARY_KEY_TP],
        counts[SUMMARY_KEY_FP],
        counts[SUMMARY_KEY_FN],
        counts[SUMMARY_KEY_TN],
    )


def detect_voice(signal: np.ndarray, morphology_mask=MORPHOLOGY_MASK_SIZE) -> np.ndarray:
    """Запустить энергетический GMM-VAD с общими настройками работы."""
    return energy_gmm_vad(
        signal,
        window=FRAME_WINDOW,
        shift=FRAME_SHIFT,
        gauss_pdf=gaussian_pdf,
        n_realignment=GMM_REALIGNMENTS,
        vad_thr=VAD_THRESHOLD,
        mask_size_morph_filt=morphology_mask,
    )


def generate_demo_signal() -> tuple[np.ndarray, np.ndarray]:
    """Создать сигнал с известными речевыми и неречевыми участками."""
    rng = np.random.default_rng(DEMO_RANDOM_SEED)
    sample_count = int(DEMO_DURATION_SECONDS * TARGET_SAMPLE_RATE)
    time = np.arange(sample_count) / TARGET_SAMPLE_RATE
    signal = rng.normal(0.0, DEMO_SILENCE_NOISE_LEVEL, sample_count)
    reference = np.zeros(sample_count, dtype=np.float32)

    for interval_index, (start_seconds, end_seconds) in enumerate(
        DEMO_SPEECH_INTERVALS
    ):
        start = int(start_seconds * TARGET_SAMPLE_RATE)
        end = int(end_seconds * TARGET_SAMPLE_RATE)
        interval_time = time[start:end]
        base_frequency = DEMO_SPEECH_FREQUENCIES[
            interval_index % len(DEMO_SPEECH_FREQUENCIES)
        ]
        envelope = np.sin(np.linspace(0.0, np.pi, end - start)) ** 0.25
        speech = sum(
            np.sin(2.0 * np.pi * base_frequency * harmonic * interval_time)
            / harmonic
            for harmonic in range(1, 5)
        )
        speech += rng.normal(0.0, DEMO_SPEECH_NOISE_LEVEL, end - start)
        signal[start:end] += 0.35 * envelope * speech
        reference[start:end] = 1.0
    return normalize_signal(signal).astype(np.float32), reference


@lru_cache(maxsize=1)
def load_impulse_response() -> np.ndarray:
    """Загрузить и при необходимости передискретизировать импульсный отклик."""
    return load_audio(IMPULSE_RESPONSE_PATH)


def plot_vad_markup(
    signal: np.ndarray,
    reference: np.ndarray,
    raw_hypothesis: np.ndarray,
    filtered_hypothesis: np.ndarray,
    output_path: Path,
) -> None:
    """Сохранить сравнение эталонной и системной VAD-разметок."""
    time = np.arange(signal.size) / TARGET_SAMPLE_RATE
    figure, axis = plt.subplots(figsize=(15, 5), constrained_layout=True)
    axis.plot(time, signal, color="tab:green", linewidth=0.6, label=LABEL_SIGNAL)
    axis.plot(time, reference, color="tab:red", label=LABEL_IDEAL_VAD)
    axis.plot(
        time,
        raw_hypothesis,
        color="tab:orange",
        alpha=0.8,
        label=LABEL_RAW_VAD,
    )
    axis.plot(
        time,
        filtered_hypothesis,
        color="tab:blue",
        label=LABEL_FILTERED_VAD,
    )
    axis.set(
        title=TITLE_VAD_MARKUP,
        xlabel=LABEL_TIME_SECONDS,
        ylabel=LABEL_AMPLITUDE,
    )
    axis.grid(True)
    axis.legend(loc="upper right")
    figure.savefig(output_path, dpi=PLOT_DPI)
    plt.close(figure)


def plot_energy_distribution(signal: np.ndarray, output_path: Path) -> None:
    """Сохранить гистограмму энергии и плотности компонентов GMM."""
    frames = framing(signal.astype(np.float64) ** 2, FRAME_WINDOW, FRAME_SHIFT)
    normalized_energy = norm_energy(frame_energy(frames))
    weights, means, sigmas = gmm_train(
        normalized_energy,
        gaussian_pdf,
        GMM_REALIGNMENTS,
    )
    x_values = np.linspace(
        normalized_energy.min(),
        normalized_energy.max(),
        1_000,
    )

    figure, axis = plt.subplots(figsize=(11, 5), constrained_layout=True)
    axis.hist(
        normalized_energy,
        bins=max(20, int(np.sqrt(normalized_energy.size))),
        density=True,
        histtype="step",
        color="black",
        label=TITLE_ENERGY_HISTOGRAM,
    )
    for component, (weight, mean, sigma) in enumerate(
        zip(weights, means, sigmas),
        start=1,
    ):
        axis.plot(
            x_values,
            weight * gaussian_pdf(x_values, mean, sigma),
            label=GMM_COMPONENT_LABEL_TEMPLATE.format(index=component),
        )
    axis.set(
        title=TITLE_ENERGY_HISTOGRAM,
        xlabel=LABEL_NORMALIZED_ENERGY,
        ylabel=LABEL_DENSITY,
    )
    axis.grid(True)
    axis.legend()
    figure.savefig(output_path, dpi=PLOT_DPI)
    plt.close(figure)


def plot_metrics(results: dict, output_path: Path) -> None:
    """Сохранить сравнение метрик для чистого и искажённых сигналов."""
    names = list(results)
    metric_specs = (
        (SUMMARY_KEY_FNR, LABEL_FNR),
        (SUMMARY_KEY_FPR, LABEL_FPR),
        (SUMMARY_KEY_PRECISION, LABEL_PRECISION),
        (SUMMARY_KEY_RECALL, LABEL_RECALL),
    )
    positions = np.arange(len(names))
    width = 0.2
    figure, axis = plt.subplots(figsize=(15, 6), constrained_layout=True)
    for index, (key, label) in enumerate(metric_specs):
        offset = (index - 1.5) * width
        axis.bar(
            positions + offset,
            [results[name][key] for name in names],
            width,
            label=label,
        )
    axis.set(
        title=TITLE_DISTORTION_METRICS,
        ylabel=LABEL_METRIC_VALUE,
        xticks=positions,
        xticklabels=names,
        ylim=(0.0, 1.05),
    )
    axis.grid(True, axis="y")
    axis.legend()
    figure.savefig(output_path, dpi=PLOT_DPI)
    plt.close(figure)


def build_distorted_signals(signal: np.ndarray) -> dict[str, np.ndarray]:
    """Сформировать чистую, реверберированную и зашумлённые версии сигнала."""
    rng = np.random.default_rng(DEMO_RANDOM_SEED)
    signals = {
        AUGMENTATION_CLEAN: signal,
        AUGMENTATION_REVERB: reverb(signal, load_impulse_response()),
    }
    for sigma in AWGN_LEVELS:
        name = AUGMENTATION_AWGN_TEMPLATE.format(sigma=sigma)
        signals[name] = awgn(signal, sigma, rng=rng)
    return signals


def save_summary(mode: str, results: dict, output_path: Path) -> None:
    """Сохранить настройки и метрики эксперимента в JSON."""
    summary = {
        SUMMARY_KEY_MODE: mode,
        SUMMARY_KEY_PARAMETERS: {
            SUMMARY_KEY_WINDOW: FRAME_WINDOW,
            SUMMARY_KEY_SHIFT: FRAME_SHIFT,
            SUMMARY_KEY_REALIGNMENTS: GMM_REALIGNMENTS,
            SUMMARY_KEY_THRESHOLD: VAD_THRESHOLD,
            SUMMARY_KEY_MORPHOLOGY: MORPHOLOGY_MASK_SIZE,
        },
        SUMMARY_KEY_RESULTS: results,
    }
    with output_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, ensure_ascii=False, indent=2)


def print_metrics(results: dict) -> None:
    """Вывести метрики всех вариантов сигнала на русском языке."""
    for name, metrics in results.items():
        print(
            METRICS_MESSAGE.format(
                name=name,
                fnr=metrics[SUMMARY_KEY_FNR],
                fpr=metrics[SUMMARY_KEY_FPR],
                precision=metrics[SUMMARY_KEY_PRECISION],
                recall=metrics[SUMMARY_KEY_RECALL],
            )
        )


def run_demo(output_dir: Path) -> None:
    """Выполнить воспроизводимый эксперимент на синтетическом сигнале."""
    output_dir.mkdir(parents=True, exist_ok=True)
    signal, reference = generate_demo_signal()
    raw_hypothesis = detect_voice(signal, morphology_mask=1)
    filtered_hypothesis = detect_voice(signal)
    plot_vad_markup(
        signal,
        reference,
        raw_hypothesis,
        filtered_hypothesis,
        output_dir / VAD_PLOT_FILENAME,
    )
    plot_energy_distribution(signal, output_dir / ENERGY_PLOT_FILENAME)

    results = {}
    for name, current_signal in build_distorted_signals(signal).items():
        hypothesis = detect_voice(current_signal)
        results[name] = calculate_metrics(reference, hypothesis)
    plot_metrics(results, output_dir / METRICS_PLOT_FILENAME)
    save_summary(
        MODE_SYNTHETIC_DEMO,
        results,
        output_dir / SUMMARY_FILENAME,
    )
    print_metrics(results)


def audio_path_from_rttm(rttm_path: Path, voxceleb_dir: Path) -> Path:
    """Преобразовать имя RTTM-файла в путь к записи VoxCeleb."""
    speaker_id, remainder = rttm_path.stem.split("_", maxsplit=1)
    video_id, file_id = remainder.rsplit("_", maxsplit=1)
    return voxceleb_dir / speaker_id / video_id / f"{file_id}.wav"


def run_voxceleb(voxceleb_dir: Path, rttm_dir: Path, output_dir: Path) -> bool:
    """Выполнить эксперимент для всех доступных RTTM-файлов VoxCeleb."""
    rttm_paths = sorted(rttm_dir.glob("*.rttm"))
    if not rttm_paths:
        print(ERROR_NO_RTTM_FILES.format(path=rttm_dir), file=sys.stderr)
        return False

    audio_paths = [audio_path_from_rttm(path, voxceleb_dir) for path in rttm_paths]
    missing = [path for path in audio_paths if not path.is_file()]
    if missing:
        examples = "\n".join(f"  - {path}" for path in missing[:3])
        print(ERROR_AUDIO_MISSING.format(examples=examples), file=sys.stderr)
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    condition_counts = {}
    first_example = None
    for rttm_path, audio_path in zip(rttm_paths, audio_paths):
        signal = load_audio(audio_path)
        reference = load_vad_markup(rttm_path, signal, TARGET_SAMPLE_RATE)
        distorted_signals = build_distorted_signals(signal)
        for name, current_signal in distorted_signals.items():
            hypothesis = detect_voice(current_signal)
            metrics = calculate_metrics(reference, hypothesis)
            condition_counts.setdefault(name, {})
            add_metric_counts(condition_counts[name], metrics)
        if first_example is None:
            first_example = (signal, reference)

    results = {
        name: finalize_metric_counts(counts)
        for name, counts in condition_counts.items()
    }
    signal, reference = first_example
    plot_vad_markup(
        signal,
        reference,
        detect_voice(signal, morphology_mask=1),
        detect_voice(signal),
        output_dir / VAD_PLOT_FILENAME,
    )
    plot_energy_distribution(signal, output_dir / ENERGY_PLOT_FILENAME)
    plot_metrics(results, output_dir / METRICS_PLOT_FILENAME)
    save_summary(MODE_VOXCELEB, results, output_dir / SUMMARY_FILENAME)
    print_metrics(results)
    return True


def parse_args() -> argparse.Namespace:
    """Прочитать параметры командной строки."""
    parser = RussianArgumentParser(
        description=CLI_DESCRIPTION,
        add_help=False,
    )
    parser._optionals.title = CLI_OPTIONS_TITLE
    parser.add_argument("-h", "--help", action="help", help=CLI_HELP_HELP)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        metavar=CLI_DIRECTORY_METAVAR,
        help=CLI_OUTPUT_DIR_HELP,
    )
    parser.add_argument(
        "--voxceleb-dir",
        type=Path,
        default=DEFAULT_VOXCELEB_DIR,
        metavar=CLI_DIRECTORY_METAVAR,
        help=CLI_VOXCELEB_DIR_HELP,
    )
    parser.add_argument(
        "--rttm-dir",
        type=Path,
        default=DEFAULT_RTTM_DIR,
        metavar=CLI_DIRECTORY_METAVAR,
        help=CLI_RTTM_DIR_HELP,
    )
    parser.add_argument("--demo", action="store_true", help=CLI_DEMO_HELP)
    return parser.parse_args()


def main() -> None:
    """Точка входа в сценарий лабораторной работы."""
    configure_console_encoding()
    args = parse_args()
    if args.demo:
        run_demo(args.output_dir)
        print(DEMO_COMPLETE_MESSAGE.format(path=args.output_dir.resolve()))
        return

    completed = run_voxceleb(
        args.voxceleb_dir.resolve(),
        args.rttm_dir.resolve(),
        args.output_dir.resolve(),
    )
    if not completed:
        raise SystemExit(1)
    print(LAB_COMPLETE_MESSAGE.format(path=args.output_dir.resolve()))


if __name__ == "__main__":
    main()
