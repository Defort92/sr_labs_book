"""Запуск четвёртой лабораторной работы и сохранение результатов."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import compute_scores, extract_features, test_dataset_loader
from constants import (
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
    CRITERION_BAYES,
    CRITERION_MAP,
    CRITERION_MINIMAX,
    CRITERION_NEYMAN_PEARSON,
    DEFAULT_COST_CORRECT_IMPOSTOR,
    DEFAULT_COST_CORRECT_TARGET,
    DEFAULT_COST_FALSE_ACCEPT,
    DEFAULT_COST_MISSED_TARGET,
    DEFAULT_MODEL_DIR,
    DEFAULT_NEYMAN_PEARSON_FNR,
    DEFAULT_PROTOCOL_PATH,
    DEFAULT_RESULTS_DIR,
    DEFAULT_TARGET_PRIOR,
    DEFAULT_VOXCELEB_DIR,
    DEMO_COMPLETE_MESSAGE,
    DEMO_IMPOSTOR_COUNT,
    DEMO_IMPOSTOR_MEAN,
    DEMO_IMPOSTOR_STD,
    DEMO_RANDOM_SEED,
    DEMO_TARGET_COUNT,
    DEMO_TARGET_MEAN,
    DEMO_TARGET_STD,
    EER_RESULT_MESSAGE,
    EMBEDDING_SIZE,
    ENCODER_ATTENTIVE_STATISTICS_POOLING,
    ERROR_DATA_MISSING,
    ERRORS_PLOT_FILENAME,
    EVALUATION_FRAMES,
    EVALUATION_SEGMENTS,
    EVALUATION_WORKERS,
    HISTOGRAM_PLOT_FILENAME,
    IMPOSTOR_LABEL,
    LABEL_DENSITY,
    LABEL_ERROR_RATE,
    LABEL_FNR,
    LABEL_FPR,
    LABEL_IMPOSTOR_DENSITY,
    LABEL_IMPOSTOR_SCORES,
    LABEL_LLR,
    LABEL_SCORE,
    LABEL_TARGET_DENSITY,
    LABEL_TARGET_SCORES,
    LABEL_THRESHOLD,
    LAB_COMPLETE_MESSAGE,
    LLR_PLOT_FILENAME,
    MEL_FILTERS,
    MINIMAX_PRIOR_STEPS,
    MINIMAX_RESULT_MESSAGE,
    MODE_DEMO,
    MODE_VOXCELEB,
    NEYMANN_PEARSON_RESULT_MESSAGE,
    PLOT_DPI,
    PRETRAINED_MODEL_PATH,
    RESULT_MESSAGE,
    SUMMARY_FILENAME,
    SUMMARY_KEY_EER,
    SUMMARY_KEY_EER_THRESHOLD,
    SUMMARY_KEY_FNR_LIMIT,
    SUMMARY_KEY_FNR,
    SUMMARY_KEY_FPR,
    SUMMARY_KEY_IMPOSTOR_COUNT,
    SUMMARY_KEY_MODE,
    SUMMARY_KEY_PARAMETERS,
    SUMMARY_KEY_PRIOR,
    SUMMARY_KEY_RESULTS,
    SUMMARY_KEY_RISK,
    SUMMARY_KEY_TARGET_COUNT,
    SUMMARY_KEY_TARGET_PRIOR,
    SUMMARY_KEY_THRESHOLD,
    TARGET_LABEL,
    TEXT_ENCODING,
    TITLE_LLR,
    TITLE_SCORE_DISTRIBUTIONS,
    TITLE_THRESHOLD_ERRORS,
)
from exercises_blank import (
    bayes_test,
    decision_curve,
    equal_error_rate,
    gaussian_pdf,
    llr,
    map_test,
    minmax_test,
    neyman_pearson_test,
    tar_imp_hists,
)
from LoadModel import load_model
from ResNetSE34V2 import MainModel


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


def generate_demo_scores():
    """Сформировать частично перекрывающиеся распределения оценок."""
    random_generator = np.random.default_rng(DEMO_RANDOM_SEED)
    target_scores = random_generator.normal(
        DEMO_TARGET_MEAN,
        DEMO_TARGET_STD,
        DEMO_TARGET_COUNT,
    )
    impostor_scores = random_generator.normal(
        DEMO_IMPOSTOR_MEAN,
        DEMO_IMPOSTOR_STD,
        DEMO_IMPOSTOR_COUNT,
    )
    all_scores = np.concatenate((target_scores, impostor_scores))
    all_labels = np.concatenate(
        (
            np.full(target_scores.size, TARGET_LABEL),
            np.full(impostor_scores.size, IMPOSTOR_LABEL),
        )
    )
    order = random_generator.permutation(all_scores.size)
    return all_scores[order], all_labels[order]


def _criterion_result(threshold, fnr, fpr, risk=None, prior=None):
    result = {
        SUMMARY_KEY_THRESHOLD: float(threshold),
        SUMMARY_KEY_FNR: float(fnr),
        SUMMARY_KEY_FPR: float(fpr),
    }
    if risk is not None:
        result[SUMMARY_KEY_RISK] = float(risk)
    if prior is not None:
        result[SUMMARY_KEY_PRIOR] = float(prior)
    return result


def analyze_scores(all_scores, all_labels, output_dir, mode):
    """Вычислить все критерии, сохранить графики и вернуть сводку."""
    output_dir.mkdir(parents=True, exist_ok=True)
    target_scores, impostor_scores = tar_imp_hists(all_scores, all_labels)
    (
        sorted_ground_truth,
        sorted_scores,
        target_density,
        impostor_density,
        log_likelihood_ratio,
    ) = llr(
        all_scores,
        all_labels,
        target_scores,
        impostor_scores,
        gaussian_pdf,
    )

    map_result = map_test(
        sorted_ground_truth,
        log_likelihood_ratio,
        target_scores,
        impostor_scores,
        DEFAULT_TARGET_PRIOR,
    )
    neyman_pearson_result = neyman_pearson_test(
        sorted_ground_truth,
        log_likelihood_ratio,
        target_scores,
        impostor_scores,
        DEFAULT_NEYMAN_PEARSON_FNR,
    )
    bayes_result = bayes_test(
        sorted_ground_truth,
        log_likelihood_ratio,
        target_scores,
        impostor_scores,
        DEFAULT_TARGET_PRIOR,
        DEFAULT_COST_CORRECT_TARGET,
        DEFAULT_COST_MISSED_TARGET,
        DEFAULT_COST_FALSE_ACCEPT,
        DEFAULT_COST_CORRECT_IMPOSTOR,
    )
    target_prior_grid = np.linspace(0.0, 1.0, MINIMAX_PRIOR_STEPS)
    minimax_result = minmax_test(
        sorted_ground_truth,
        log_likelihood_ratio,
        target_scores,
        impostor_scores,
        target_prior_grid,
        DEFAULT_COST_CORRECT_TARGET,
        DEFAULT_COST_MISSED_TARGET,
        DEFAULT_COST_FALSE_ACCEPT,
        DEFAULT_COST_CORRECT_IMPOSTOR,
    )
    eer, eer_threshold = equal_error_rate(
        np.asarray(all_labels, dtype=bool),
        np.asarray(all_scores, dtype=np.float64),
    )

    print(
        RESULT_MESSAGE.format(
            criterion=CRITERION_MAP,
            threshold=map_result[0],
            fnr=map_result[1],
            fpr=map_result[2],
            risk=map_result[3],
        )
    )
    print(
        NEYMANN_PEARSON_RESULT_MESSAGE.format(
            criterion=CRITERION_NEYMAN_PEARSON,
            threshold=neyman_pearson_result[0],
            fnr=DEFAULT_NEYMAN_PEARSON_FNR,
            fpr=neyman_pearson_result[1],
        )
    )
    print(
        RESULT_MESSAGE.format(
            criterion=CRITERION_BAYES,
            threshold=bayes_result[0],
            fnr=bayes_result[1],
            fpr=bayes_result[2],
            risk=bayes_result[3],
        )
    )
    print(
        MINIMAX_RESULT_MESSAGE.format(
            criterion=CRITERION_MINIMAX,
            threshold=minimax_result[0],
            fnr=minimax_result[1],
            fpr=minimax_result[2],
            risk=minimax_result[3],
            prior=minimax_result[4],
        )
    )
    print(EER_RESULT_MESSAGE.format(eer=eer))

    _save_plots(
        output_dir,
        target_scores,
        impostor_scores,
        sorted_scores,
        target_density,
        impostor_density,
        log_likelihood_ratio,
        sorted_ground_truth,
    )

    summary = {
        SUMMARY_KEY_MODE: mode,
        SUMMARY_KEY_PARAMETERS: {
            SUMMARY_KEY_TARGET_PRIOR: DEFAULT_TARGET_PRIOR,
            SUMMARY_KEY_FNR_LIMIT: DEFAULT_NEYMAN_PEARSON_FNR,
            SUMMARY_KEY_TARGET_COUNT: int(target_scores.size),
            SUMMARY_KEY_IMPOSTOR_COUNT: int(impostor_scores.size),
        },
        SUMMARY_KEY_RESULTS: {
            CRITERION_MAP: _criterion_result(*map_result),
            CRITERION_NEYMAN_PEARSON: _criterion_result(
                neyman_pearson_result[0],
                DEFAULT_NEYMAN_PEARSON_FNR,
                neyman_pearson_result[1],
            ),
            CRITERION_BAYES: _criterion_result(*bayes_result),
            CRITERION_MINIMAX: _criterion_result(*minimax_result),
            SUMMARY_KEY_EER: eer,
            SUMMARY_KEY_EER_THRESHOLD: eer_threshold,
        },
    }
    with (output_dir / SUMMARY_FILENAME).open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    return summary


def _save_plots(
    output_dir,
    target_scores,
    impostor_scores,
    sorted_scores,
    target_density,
    impostor_density,
    log_likelihood_ratio,
    sorted_ground_truth,
):
    bins = max(10, int(np.sqrt(min(target_scores.size, impostor_scores.size))))
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.hist(
        target_scores,
        bins=bins,
        histtype="step",
        density=True,
        linewidth=2,
        label=LABEL_TARGET_SCORES,
    )
    axis.hist(
        impostor_scores,
        bins=bins,
        histtype="step",
        density=True,
        linewidth=2,
        label=LABEL_IMPOSTOR_SCORES,
    )
    axis.set_title(TITLE_SCORE_DISTRIBUTIONS)
    axis.set_xlabel(LABEL_SCORE)
    axis.set_ylabel(LABEL_DENSITY)
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / HISTOGRAM_PLOT_FILENAME, dpi=PLOT_DPI)
    plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(sorted_scores, target_density, label=LABEL_TARGET_DENSITY)
    axes[0].plot(sorted_scores, impostor_density, label=LABEL_IMPOSTOR_DENSITY)
    axes[0].set_ylabel(LABEL_DENSITY)
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].plot(sorted_scores, log_likelihood_ratio)
    axes[1].axhline(0.0, color="black", linestyle="--")
    axes[1].set_xlabel(LABEL_SCORE)
    axes[1].set_ylabel(LABEL_LLR)
    axes[1].grid(alpha=0.3)
    figure.suptitle(TITLE_LLR)
    figure.tight_layout()
    figure.savefig(output_dir / LLR_PLOT_FILENAME, dpi=PLOT_DPI)
    plt.close(figure)

    thresholds, false_negative_rates, false_positive_rates = decision_curve(
        sorted_ground_truth,
        log_likelihood_ratio,
    )
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(thresholds, false_negative_rates, label=LABEL_FNR)
    axis.plot(thresholds, false_positive_rates, label=LABEL_FPR)
    axis.set_title(TITLE_THRESHOLD_ERRORS)
    axis.set_xlabel(LABEL_THRESHOLD)
    axis.set_ylabel(LABEL_ERROR_RATE)
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / ERRORS_PLOT_FILENAME, dpi=PLOT_DPI)
    plt.close(figure)


def run_demo(output_dir):
    all_scores, all_labels = generate_demo_scores()
    analyze_scores(all_scores, all_labels, output_dir, MODE_DEMO)


def run_full(output_dir):
    for required_path in (
        DEFAULT_PROTOCOL_PATH,
        DEFAULT_VOXCELEB_DIR,
        PRETRAINED_MODEL_PATH,
    ):
        if not required_path.exists():
            raise FileNotFoundError(ERROR_DATA_MISSING.format(path=required_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MainModel(
        nOut=EMBEDDING_SIZE,
        encoder_type=ENCODER_ATTENTIVE_STATISTICS_POOLING,
        n_mels=MEL_FILTERS,
    ).to(device)
    model = load_model(model, [], DEFAULT_MODEL_DIR)
    model.eval()

    with DEFAULT_PROTOCOL_PATH.open("r", encoding=TEXT_ENCODING) as protocol_file:
        protocol_lines = protocol_file.readlines()
    files = sorted(
        set(
            itertools.chain.from_iterable(
                line.strip().split()[-2:] for line in protocol_lines
            )
        )
    )
    dataset = test_dataset_loader(
        files,
        test_path=str(DEFAULT_VOXCELEB_DIR),
        eval_frames=EVALUATION_FRAMES,
        num_eval=EVALUATION_SEGMENTS,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=EVALUATION_WORKERS,
    )
    features = extract_features(model, loader)
    all_scores, all_labels, _ = compute_scores(features, protocol_lines)
    analyze_scores(all_scores, all_labels, output_dir, MODE_VOXCELEB)


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
            print(DEMO_COMPLETE_MESSAGE.format(path=arguments.output_dir.resolve()))
        else:
            run_full(arguments.output_dir)
            print(LAB_COMPLETE_MESSAGE.format(path=arguments.output_dir.resolve()))
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
