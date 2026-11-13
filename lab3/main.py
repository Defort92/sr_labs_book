"""Запуск третьей лабораторной работы и сохранение результатов."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import fftconvolve
from torch.utils.data import DataLoader, Dataset

from constants import (
    ANGULAR_MARGIN,
    ANGULAR_SCALE,
    CLI_DEMO_HELP,
    CLI_DESCRIPTION,
    CLI_DIRECTORY_METAVAR,
    CLI_HELP_HELP,
    CLI_OPTIONS_PREFIX,
    CLI_OPTIONS_TITLE,
    CLI_OUTPUT_DIR_HELP,
    CLI_RUSSIAN_OPTIONS_PREFIX,
    CLI_RUSSIAN_USAGE_PREFIX,
    CLI_USAGE_PREFIX,
    COMPARISON_PLOT_FILENAME,
    DEFAULT_AUGMENTED_CHECKPOINT_DIR,
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_RESULTS_DIR,
    DEFAULT_VOXCELEB_DEV_DIR,
    DEFAULT_VOXCELEB_TEST_DIR,
    DEMO_TEST_RESULT_MESSAGE,
    DEMO_AUDIO_SECONDS,
    DEMO_AUGMENTED_TRAIN_NOISE,
    DEMO_BASE_FREQUENCIES,
    DEMO_BATCH_SIZE,
    DEMO_EMBEDDING_SIZE,
    DEMO_EPOCHS,
    DEMO_FILTERS,
    DEMO_LAYERS,
    DEMO_LEARNING_RATE,
    DEMO_MAX_THREADS,
    DEMO_MEL_FILTERS,
    DEMO_RANDOM_SEED,
    DEMO_SPEAKER_COUNT,
    DEMO_TEST_EXAMPLES_PER_SPEAKER,
    DEMO_TEST_NOISE,
    DEMO_TRAIN_EXAMPLES_PER_SPEAKER,
    DEMO_TRAIN_NOISE,
    EMBEDDING_SIZE,
    ENCODER_STATISTICAL_POOLING,
    EPOCH_RESULT_MESSAGE,
    ERROR_DATASET_MISSING,
    EVAL_BATCH_SIZE,
    EVAL_MAX_FRAMES,
    EVAL_WORKERS,
    IDENTIFICATION_PROTOCOL_PATH,
    LABEL_ACCURACY_PERCENT,
    LABEL_EPOCH,
    LABEL_LOSS,
    LABEL_TRAIN_LOSS_WITH_AUGMENTATION,
    LABEL_TRAIN_LOSS_WITHOUT_AUGMENTATION,
    LEARNING_RATE,
    MAX_EPOCHS,
    MEL_FILTERS,
    MODEL_WITH_AUGMENTATION,
    MODEL_WITHOUT_AUGMENTATION,
    MUSAN_PATH,
    ONE_CYCLE_DIV_FACTOR,
    ONE_CYCLE_FINAL_DIV_FACTOR,
    ONE_CYCLE_START_FRACTION,
    PLOT_DPI,
    RESNET_FILTERS,
    RESNET_LAYERS,
    RIR_PATH,
    SAMPLE_RATE,
    SUMMARY_FILENAME,
    SUMMARY_KEY_HISTORY,
    SUMMARY_KEY_PARAMETERS,
    SUMMARY_KEY_RESULTS,
    TEST_CLEAN,
    TEST_NOISY,
    TITLE_ACCURACY_COMPARISON,
    TITLE_TRAINING_HISTORY,
    TRAIN_BATCH_SIZE,
    TRAIN_MAX_FRAMES,
    TRAIN_WORKERS,
    TRAINING_PLOT_FILENAME,
    VALIDATION_INTERVAL,
    VALIDATION_RESULT_MESSAGE,
    WEIGHT_DECAY,
    DEMO_COMPLETE_MESSAGE,
    LAB_COMPLETE_MESSAGE,
)
from exercises_blank import (
    MainModel,
    ResNet,
    test_dataset_loader,
    test_network,
    train_dataset_loader,
    train_network,
)
from load_save_pth import saveParameters
from LossFunction import AAMSoftmaxLoss
from Optimizer import SGDOptimizer
from ResNetBlocks import BasicBlock
from Scheduler import OneCycleLRScheduler, StepLRScheduler


def configure_console_encoding():
    """Настроить корректный вывод русского текста в Windows."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


class RussianArgumentParser(argparse.ArgumentParser):
    """Парсер аргументов с русскими заголовками справки."""

    def format_usage(self):
        return super().format_usage().replace(
            CLI_USAGE_PREFIX,
            CLI_RUSSIAN_USAGE_PREFIX,
            1,
        )

    def format_help(self):
        return (
            super()
            .format_help()
            .replace(CLI_USAGE_PREFIX, CLI_RUSSIAN_USAGE_PREFIX, 1)
            .replace(CLI_OPTIONS_PREFIX, CLI_RUSSIAN_OPTIONS_PREFIX, 1)
        )


class SpeakerTensorDataset(Dataset):
    """Небольшой набор синтетических записей для демонстрации."""

    def __init__(self, signals, labels):
        self.signals = torch.as_tensor(signals, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __getitem__(self, index):
        return self.signals[index].unsqueeze(0), self.labels[index]

    def __len__(self):
        return len(self.labels)


def set_random_seed(seed):
    """Зафиксировать генераторы случайных чисел."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def generate_speaker_signal(frequency, noise_level, rng, augmented=False):
    """Сформировать короткий сигнал с устойчивыми признаками диктора."""
    sample_count = int(SAMPLE_RATE * DEMO_AUDIO_SECONDS)
    time = np.arange(sample_count, dtype=np.float64) / SAMPLE_RATE
    phase = rng.uniform(0.0, 2.0 * np.pi)
    second_phase = rng.uniform(0.0, 2.0 * np.pi)
    signal = 0.70 * np.sin(2.0 * np.pi * frequency * time + phase)
    signal += 0.25 * np.sin(
        2.0 * np.pi * frequency * 2.0 * time + second_phase
    )
    signal += 0.10 * np.sin(2.0 * np.pi * frequency * 3.0 * time)
    envelope_frequency = rng.uniform(1.5, 3.5)
    envelope = 0.75 + 0.25 * np.sin(
        2.0 * np.pi * envelope_frequency * time + phase
    )
    signal *= envelope

    if augmented and rng.random() < 0.4:
        impulse_response = np.zeros(300, dtype=np.float64)
        impulse_response[0] = 1.0
        impulse_response[rng.integers(45, 90)] = rng.uniform(0.15, 0.30)
        impulse_response[rng.integers(120, 220)] = rng.uniform(0.05, 0.15)
        impulse_response /= np.linalg.norm(impulse_response)
        signal = fftconvolve(signal, impulse_response, mode="full")[:sample_count]

    signal += rng.normal(0.0, noise_level, sample_count)
    peak = max(np.max(np.abs(signal)), np.finfo(np.float32).eps)
    return (signal / peak).astype(np.float32)


def generate_demo_dataset(examples_per_speaker, noise_level, seed, augmented):
    """Создать сбалансированный набор синтетических дикторов."""
    rng = np.random.default_rng(seed)
    signals = []
    labels = []
    for speaker_index in range(DEMO_SPEAKER_COUNT):
        frequency = DEMO_BASE_FREQUENCIES[speaker_index]
        for _ in range(examples_per_speaker):
            current_noise = noise_level
            if augmented:
                current_noise = rng.uniform(DEMO_TRAIN_NOISE, noise_level)
            signals.append(
                generate_speaker_signal(
                    frequency,
                    current_noise,
                    rng,
                    augmented=augmented,
                )
            )
            labels.append(speaker_index)
    return SpeakerTensorDataset(np.stack(signals), labels)


def build_model(
    speaker_count,
    layers,
    filters,
    mel_filters,
    embedding_size,
    device,
):
    """Создать модель и функцию потерь AAM-Softmax."""
    speaker_model = ResNet(
        BasicBlock,
        layers=layers,
        activation=nn.ReLU,
        num_filters=filters,
        nOut=embedding_size,
        encoder_type=ENCODER_STATISTICAL_POOLING,
        n_mels=mel_filters,
        log_input=True,
    )
    loss_function = AAMSoftmaxLoss(
        nOut=embedding_size,
        nClasses=speaker_count,
        margin=ANGULAR_MARGIN,
        scale=ANGULAR_SCALE,
    )
    return MainModel(speaker_model, loss_function).to(device)


def train_demo_model(train_loader, device, seed):
    """Обучить маленькую ResNet на демонстрационном наборе."""
    set_random_seed(seed)
    model = build_model(
        DEMO_SPEAKER_COUNT,
        DEMO_LAYERS,
        DEMO_FILTERS,
        DEMO_MEL_FILTERS,
        DEMO_EMBEDDING_SIZE,
        device,
    )
    optimizer = SGDOptimizer(
        model.parameters(),
        lr=DEMO_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = StepLRScheduler(
        optimizer,
        test_interval=1,
        lr_decay=0.75,
    )
    history = []
    for epoch in range(DEMO_EPOCHS):
        loss, accuracy = train_network(
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch + 1,
            verbose=False,
        )
        history.append({"эпоха": epoch + 1, "потери": loss, "точность": accuracy})
        print(
            EPOCH_RESULT_MESSAGE.format(
                epoch=epoch + 1,
                loss=loss,
                accuracy=accuracy,
            )
        )
    return model, optimizer, scheduler, history


def evaluate_model(model, clean_loader, noisy_loader):
    """Оценить одну модель на чистых и зашумлённых записях."""
    _, clean_accuracy = test_network(clean_loader, model)
    _, noisy_accuracy = test_network(noisy_loader, model)
    return {TEST_CLEAN: clean_accuracy, TEST_NOISY: noisy_accuracy}


def save_demo_plots(output_dir, clean_history, augmented_history, results):
    """Сохранить графики обучения и сравнения моделей."""
    epochs = [item["эпоха"] for item in clean_history]
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(
        epochs,
        [item["потери"] for item in clean_history],
        marker="o",
        label=LABEL_TRAIN_LOSS_WITHOUT_AUGMENTATION,
    )
    axis.plot(
        epochs,
        [item["потери"] for item in augmented_history],
        marker="o",
        label=LABEL_TRAIN_LOSS_WITH_AUGMENTATION,
    )
    axis.set_title(TITLE_TRAINING_HISTORY)
    axis.set_xlabel(LABEL_EPOCH)
    axis.set_ylabel(LABEL_LOSS)
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / TRAINING_PLOT_FILENAME, dpi=PLOT_DPI)
    plt.close(figure)

    model_names = [MODEL_WITHOUT_AUGMENTATION, MODEL_WITH_AUGMENTATION]
    positions = np.arange(len(model_names))
    width = 0.35
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(
        positions - width / 2,
        [results[name][TEST_CLEAN] for name in model_names],
        width,
        label=TEST_CLEAN,
    )
    axis.bar(
        positions + width / 2,
        [results[name][TEST_NOISY] for name in model_names],
        width,
        label=TEST_NOISY,
    )
    axis.set_xticks(positions, model_names)
    axis.set_ylim(0.0, 105.0)
    axis.set_title(TITLE_ACCURACY_COMPARISON)
    axis.set_ylabel(LABEL_ACCURACY_PERCENT)
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / COMPARISON_PLOT_FILENAME, dpi=PLOT_DPI)
    plt.close(figure)


def write_summary(output_dir, summary):
    """Сохранить результаты в JSON с русскими ключами."""
    with (output_dir / SUMMARY_FILENAME).open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, ensure_ascii=False, indent=2)


def run_demo(output_dir):
    """Провести быстрый эксперимент без внешних наборов данных."""
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(DEMO_MAX_THREADS)
    set_random_seed(DEMO_RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clean_train_dataset = generate_demo_dataset(
        DEMO_TRAIN_EXAMPLES_PER_SPEAKER,
        DEMO_TRAIN_NOISE,
        DEMO_RANDOM_SEED,
        augmented=False,
    )
    augmented_train_dataset = generate_demo_dataset(
        DEMO_TRAIN_EXAMPLES_PER_SPEAKER,
        DEMO_AUGMENTED_TRAIN_NOISE,
        DEMO_RANDOM_SEED + 1,
        augmented=True,
    )
    clean_test_dataset = generate_demo_dataset(
        DEMO_TEST_EXAMPLES_PER_SPEAKER,
        DEMO_TRAIN_NOISE,
        DEMO_RANDOM_SEED + 2,
        augmented=False,
    )
    noisy_test_dataset = generate_demo_dataset(
        DEMO_TEST_EXAMPLES_PER_SPEAKER,
        DEMO_TEST_NOISE,
        DEMO_RANDOM_SEED + 3,
        augmented=True,
    )

    loader_generator = torch.Generator().manual_seed(DEMO_RANDOM_SEED)
    clean_train_loader = DataLoader(
        clean_train_dataset,
        batch_size=DEMO_BATCH_SIZE,
        shuffle=True,
        generator=loader_generator,
    )
    augmented_train_loader = DataLoader(
        augmented_train_dataset,
        batch_size=DEMO_BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(DEMO_RANDOM_SEED + 1),
    )
    clean_test_loader = DataLoader(
        clean_test_dataset,
        batch_size=DEMO_BATCH_SIZE,
    )
    noisy_test_loader = DataLoader(
        noisy_test_dataset,
        batch_size=DEMO_BATCH_SIZE,
    )

    clean_model, _, _, clean_history = train_demo_model(
        clean_train_loader,
        device,
        DEMO_RANDOM_SEED,
    )
    augmented_model, _, _, augmented_history = train_demo_model(
        augmented_train_loader,
        device,
        DEMO_RANDOM_SEED,
    )

    results = {
        MODEL_WITHOUT_AUGMENTATION: evaluate_model(
            clean_model,
            clean_test_loader,
            noisy_test_loader,
        ),
        MODEL_WITH_AUGMENTATION: evaluate_model(
            augmented_model,
            clean_test_loader,
            noisy_test_loader,
        ),
    }
    for model_name, model_results in results.items():
        for test_name, accuracy in model_results.items():
            print(
                DEMO_TEST_RESULT_MESSAGE.format(
                    model_name=model_name,
                    test_name=test_name,
                    accuracy=accuracy,
                )
            )

    summary = {
        SUMMARY_KEY_PARAMETERS: {
            "число_дикторов": DEMO_SPEAKER_COUNT,
            "число_эпох": DEMO_EPOCHS,
            "размер_эмбеддинга": DEMO_EMBEDDING_SIZE,
            "число_мел_фильтров": DEMO_MEL_FILTERS,
        },
        SUMMARY_KEY_HISTORY: {
            MODEL_WITHOUT_AUGMENTATION: clean_history,
            MODEL_WITH_AUGMENTATION: augmented_history,
        },
        SUMMARY_KEY_RESULTS: results,
    }
    save_demo_plots(output_dir, clean_history, augmented_history, results)
    write_summary(output_dir, summary)


def build_protocol_lists():
    """Построить списки train/validation/test по протоколу VoxCeleb1."""
    if not IDENTIFICATION_PROTOCOL_PATH.is_file():
        raise FileNotFoundError(
            ERROR_DATASET_MISSING.format(path=IDENTIFICATION_PROTOCOL_PATH)
        )
    if not DEFAULT_VOXCELEB_DEV_DIR.is_dir():
        raise FileNotFoundError(
            ERROR_DATASET_MISSING.format(path=DEFAULT_VOXCELEB_DEV_DIR)
        )
    if not DEFAULT_VOXCELEB_TEST_DIR.is_dir():
        raise FileNotFoundError(
            ERROR_DATASET_MISSING.format(path=DEFAULT_VOXCELEB_TEST_DIR)
        )

    excluded_speakers = {path.name for path in DEFAULT_VOXCELEB_TEST_DIR.iterdir()}
    split_lists = {"1": [], "2": [], "3": []}
    speakers = set()
    with IDENTIFICATION_PROTOCOL_PATH.open("r", encoding="utf-8") as protocol_file:
        for raw_line in protocol_file:
            split_id, relative_path = raw_line.strip().split()
            speaker_id = Path(relative_path).parts[0]
            if speaker_id in excluded_speakers:
                continue
            speakers.add(speaker_id)
            split_lists[split_id].append(f"{speaker_id} {relative_path}")
    return split_lists["1"], split_lists["2"], split_lists["3"], len(speakers)


def train_full_model(train_list, validation_list, speaker_count, augment, device):
    """Обучить полную модель VoxCeleb1 с выбранным режимом аугментации."""
    model = build_model(
        speaker_count,
        RESNET_LAYERS,
        RESNET_FILTERS,
        MEL_FILTERS,
        EMBEDDING_SIZE,
        device,
    )
    train_dataset = train_dataset_loader(
        train_list,
        TRAIN_MAX_FRAMES,
        str(DEFAULT_VOXCELEB_DEV_DIR),
        augment=augment,
        musan_path=str(MUSAN_PATH),
        rir_path=str(RIR_PATH),
    )
    validation_dataset = test_dataset_loader(
        validation_list,
        EVAL_MAX_FRAMES,
        str(DEFAULT_VOXCELEB_DEV_DIR),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=TRAIN_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=EVAL_BATCH_SIZE,
        num_workers=EVAL_WORKERS,
    )
    optimizer = SGDOptimizer(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = OneCycleLRScheduler(
        optimizer,
        pct_start=ONE_CYCLE_START_FRACTION,
        cycle_momentum=False,
        max_lr=LEARNING_RATE,
        div_factor=ONE_CYCLE_DIV_FACTOR,
        final_div_factor=ONE_CYCLE_FINAL_DIV_FACTOR,
        total_steps=MAX_EPOCHS * len(train_loader),
    )
    checkpoint_dir = (
        DEFAULT_AUGMENTED_CHECKPOINT_DIR if augment else DEFAULT_CHECKPOINT_DIR
    )
    history = []
    for epoch in range(MAX_EPOCHS):
        loss, accuracy = train_network(
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch + 1,
            verbose=True,
        )
        record = {"эпоха": epoch + 1, "потери": loss, "точность": accuracy}
        if (epoch + 1) % VALIDATION_INTERVAL == 0:
            _, validation_accuracy = test_network(validation_loader, model)
            record["точность_валидации"] = validation_accuracy
            print(
                VALIDATION_RESULT_MESSAGE.format(
                    epoch=epoch + 1,
                    accuracy=validation_accuracy,
                )
            )
            saveParameters(model, optimizer, scheduler, epoch, checkpoint_dir)
        history.append(record)
    return model, history


def run_full(output_dir):
    """Выполнить полный эксперимент на VoxCeleb1."""
    output_dir.mkdir(parents=True, exist_ok=True)
    train_list, validation_list, test_list, speaker_count = build_protocol_lists()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clean_model, clean_history = train_full_model(
        train_list,
        validation_list,
        speaker_count,
        augment=False,
        device=device,
    )
    augmented_model, augmented_history = train_full_model(
        train_list,
        validation_list,
        speaker_count,
        augment=True,
        device=device,
    )
    test_dataset = test_dataset_loader(
        test_list,
        EVAL_MAX_FRAMES,
        str(DEFAULT_VOXCELEB_DEV_DIR),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=EVAL_BATCH_SIZE,
        num_workers=EVAL_WORKERS,
    )
    _, clean_accuracy = test_network(test_loader, clean_model)
    _, augmented_accuracy = test_network(test_loader, augmented_model)
    results = {
        MODEL_WITHOUT_AUGMENTATION: clean_accuracy,
        MODEL_WITH_AUGMENTATION: augmented_accuracy,
    }
    summary = {
        SUMMARY_KEY_PARAMETERS: {
            "число_дикторов": speaker_count,
            "число_эпох": MAX_EPOCHS,
            "размер_эмбеддинга": EMBEDDING_SIZE,
        },
        SUMMARY_KEY_HISTORY: {
            MODEL_WITHOUT_AUGMENTATION: clean_history,
            MODEL_WITH_AUGMENTATION: augmented_history,
        },
        SUMMARY_KEY_RESULTS: results,
    }
    write_summary(output_dir, summary)


def parse_arguments():
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
    parser.add_argument("--demo", action="store_true", help=CLI_DEMO_HELP)
    return parser.parse_args()


def main():
    configure_console_encoding()
    arguments = parse_arguments()
    try:
        if arguments.demo:
            run_demo(arguments.output_dir)
            print(
                DEMO_COMPLETE_MESSAGE.format(
                    path=arguments.output_dir.resolve()
                )
            )
        else:
            run_full(arguments.output_dir)
            print(
                LAB_COMPLETE_MESSAGE.format(
                    path=arguments.output_dir.resolve()
                )
            )
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
