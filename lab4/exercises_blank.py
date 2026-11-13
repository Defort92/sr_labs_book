"""Критерии принятия решения для четвёртой лабораторной работы."""

from __future__ import annotations

import numpy as np

from constants import (
    ERROR_CLASS_EMPTY,
    ERROR_COSTS,
    ERROR_FNR_RANGE,
    ERROR_LABEL_VALUE,
    ERROR_PRIOR_RANGE,
    ERROR_SCORES_EMPTY,
    ERROR_SCORES_LENGTH,
    ERROR_SCORES_ONE_DIMENSIONAL,
    IMPOSTOR_LABEL,
    NUMERICAL_EPSILON,
    TARGET_LABEL,
)


def _validated_scores_and_labels(all_scores, all_labels):
    """Проверить и привести оценки и метки к одномерным массивам."""
    scores = np.asarray(all_scores, dtype=np.float64)
    labels = np.asarray(all_labels)
    if scores.ndim != 1 or labels.ndim != 1:
        raise ValueError(ERROR_SCORES_ONE_DIMENSIONAL)
    if scores.size != labels.size:
        raise ValueError(ERROR_SCORES_LENGTH)
    if scores.size == 0:
        raise ValueError(ERROR_SCORES_EMPTY)
    if not np.all(np.isin(labels, (IMPOSTOR_LABEL, TARGET_LABEL))):
        raise ValueError(ERROR_LABEL_VALUE)
    labels = labels.astype(bool)
    if not np.any(labels) or np.all(labels):
        raise ValueError(ERROR_CLASS_EMPTY)
    return scores, labels


def _validate_prior(target_prior):
    if not 0.0 <= target_prior <= 1.0:
        raise ValueError(ERROR_PRIOR_RANGE)


def _validate_costs(cost_correct_target, cost_missed_target, cost_false_accept, cost_correct_impostor):
    if (
        cost_missed_target <= cost_correct_target
        or cost_false_accept <= cost_correct_impostor
    ):
        raise ValueError(ERROR_COSTS)


def gaussian_pdf(value, mean, standard_deviation):
    """Вычислить плотность одномерного нормального распределения."""
    safe_std = max(abs(float(standard_deviation)), NUMERICAL_EPSILON)
    coefficient = 1.0 / (safe_std * np.sqrt(2.0 * np.pi))
    return coefficient * np.exp(-0.5 * ((value - mean) / safe_std) ** 2)


def tar_imp_hists(all_scores, all_labels):
    """Разделить оценки на таргет- и импостор-наборы."""
    scores, labels = _validated_scores_and_labels(all_scores, all_labels)
    return scores[labels], scores[~labels]


def llr(all_scores, all_labels, tar_scores, imp_scores, gauss_pdf_function=None):
    """Оценить гауссовы плотности и логарифм отношения правдоподобия."""
    scores, labels = _validated_scores_and_labels(all_scores, all_labels)
    target_scores = np.asarray(tar_scores, dtype=np.float64)
    impostor_scores = np.asarray(imp_scores, dtype=np.float64)
    if target_scores.size == 0 or impostor_scores.size == 0:
        raise ValueError(ERROR_CLASS_EMPTY)
    if gauss_pdf_function is None:
        gauss_pdf_function = gaussian_pdf

    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    sorted_ground_truth = labels[order]

    target_density = gauss_pdf_function(
        sorted_scores,
        np.mean(target_scores),
        np.std(target_scores),
    )
    impostor_density = gauss_pdf_function(
        sorted_scores,
        np.mean(impostor_scores),
        np.std(impostor_scores),
    )
    target_density = np.maximum(target_density, NUMERICAL_EPSILON)
    impostor_density = np.maximum(impostor_density, NUMERICAL_EPSILON)
    log_likelihood_ratio = np.log(target_density) - np.log(impostor_density)
    return (
        sorted_ground_truth,
        sorted_scores,
        target_density,
        impostor_density,
        log_likelihood_ratio,
    )


def _error_rates(ground_truth, log_likelihood_ratio, threshold):
    decisions = log_likelihood_ratio > threshold
    false_negative_rate = np.mean(~decisions[ground_truth])
    false_positive_rate = np.mean(decisions[~ground_truth])
    return float(false_negative_rate), float(false_positive_rate)


def decision_curve(ground_truth_sort, LLR):
    """Вычислить FNR и FPR для всех возможных конечных порогов."""
    ground_truth = np.asarray(ground_truth_sort, dtype=bool)
    log_likelihood_ratio = np.asarray(LLR, dtype=np.float64)
    if ground_truth.ndim != 1 or log_likelihood_ratio.ndim != 1:
        raise ValueError(ERROR_SCORES_ONE_DIMENSIONAL)
    if ground_truth.size != log_likelihood_ratio.size:
        raise ValueError(ERROR_SCORES_LENGTH)
    thresholds = np.unique(np.sort(log_likelihood_ratio))
    false_negative_rates = np.empty(thresholds.size, dtype=np.float64)
    false_positive_rates = np.empty(thresholds.size, dtype=np.float64)
    for index, threshold in enumerate(thresholds):
        false_negative_rates[index], false_positive_rates[index] = _error_rates(
            ground_truth,
            log_likelihood_ratio,
            threshold,
        )
    return thresholds, false_negative_rates, false_positive_rates


def _decision_curve_with_boundaries(ground_truth, log_likelihood_ratio):
    thresholds, false_negative_rates, false_positive_rates = decision_curve(
        ground_truth,
        log_likelihood_ratio,
    )
    thresholds = np.concatenate(([-np.inf], thresholds, [np.inf]))
    false_negative_rates = np.concatenate(([0.0], false_negative_rates, [1.0]))
    false_positive_rates = np.concatenate(([1.0], false_positive_rates, [0.0]))
    return thresholds, false_negative_rates, false_positive_rates


def map_test(ground_truth_sort, LLR, tar_scores, imp_scores, P_Htar):
    """Выбрать эмпирический порог минимума вероятности ошибки."""
    _validate_prior(P_Htar)
    ground_truth = np.asarray(ground_truth_sort, dtype=bool)
    log_likelihood_ratio = np.asarray(LLR, dtype=np.float64)
    thresholds, false_negative_rates, false_positive_rates = (
        _decision_curve_with_boundaries(ground_truth, log_likelihood_ratio)
    )
    error_probabilities = (
        false_negative_rates * P_Htar
        + false_positive_rates * (1.0 - P_Htar)
    )
    best_index = int(np.argmin(error_probabilities))
    return (
        float(thresholds[best_index]),
        float(false_negative_rates[best_index]),
        float(false_positive_rates[best_index]),
        float(error_probabilities[best_index]),
    )


def neyman_pearson_test(ground_truth_sort, LLR, tar_scores, imp_scores, fnr):
    """Минимизировать FPR при ограничении на максимально допустимый FNR."""
    if not 0.0 <= fnr <= 1.0:
        raise ValueError(ERROR_FNR_RANGE)
    ground_truth = np.asarray(ground_truth_sort, dtype=bool)
    log_likelihood_ratio = np.asarray(LLR, dtype=np.float64)
    thresholds, false_negative_rates, false_positive_rates = (
        _decision_curve_with_boundaries(ground_truth, log_likelihood_ratio)
    )
    feasible = np.flatnonzero(false_negative_rates <= fnr + NUMERICAL_EPSILON)
    minimum_fpr = np.min(false_positive_rates[feasible])
    best_candidates = feasible[
        np.isclose(false_positive_rates[feasible], minimum_fpr)
    ]
    best_index = int(best_candidates[-1])
    return float(thresholds[best_index]), float(false_positive_rates[best_index])


def bayes_test(
    ground_truth_sort,
    LLR,
    tar_scores,
    imp_scores,
    P_Htar,
    C00,
    C10,
    C01,
    C11,
):
    """Применить байесовский порог и вычислить средний риск."""
    _validate_prior(P_Htar)
    _validate_costs(C00, C10, C01, C11)
    ground_truth = np.asarray(ground_truth_sort, dtype=bool)
    log_likelihood_ratio = np.asarray(LLR, dtype=np.float64)

    if P_Htar == 0.0:
        threshold = np.inf
    elif P_Htar == 1.0:
        threshold = -np.inf
    else:
        numerator = (C01 - C11) * (1.0 - P_Htar)
        denominator = (C10 - C00) * P_Htar
        threshold = float(np.log(numerator / denominator))

    false_negative_rate, false_positive_rate = _error_rates(
        ground_truth,
        log_likelihood_ratio,
        threshold,
    )
    average_cost = (
        C00 * (1.0 - false_negative_rate) * P_Htar
        + C10 * false_negative_rate * P_Htar
        + C01 * false_positive_rate * (1.0 - P_Htar)
        + C11 * (1.0 - false_positive_rate) * (1.0 - P_Htar)
    )
    return (
        float(threshold),
        false_negative_rate,
        false_positive_rate,
        float(average_cost),
    )


def minmax_test(
    ground_truth_sort,
    LLR,
    tar_scores,
    imp_scores,
    P_Htar_thr,
    C00,
    C10,
    C01,
    C11,
):
    """Найти наименее благоприятный prior и соответствующее решение Байеса."""
    target_priors = np.asarray(P_Htar_thr, dtype=np.float64)
    if target_priors.ndim != 1 or target_priors.size == 0:
        raise ValueError(ERROR_PRIOR_RANGE)
    if np.any((target_priors < 0.0) | (target_priors > 1.0)):
        raise ValueError(ERROR_PRIOR_RANGE)

    results = [
        bayes_test(
            ground_truth_sort,
            LLR,
            tar_scores,
            imp_scores,
            float(target_prior),
            C00,
            C10,
            C01,
            C11,
        )
        for target_prior in target_priors
    ]
    average_costs = np.asarray([result[3] for result in results])
    worst_prior_index = int(np.argmax(average_costs))
    threshold, fnr, fpr, average_cost = results[worst_prior_index]
    return (
        threshold,
        fnr,
        fpr,
        average_cost,
        float(target_priors[worst_prior_index]),
    )


def equal_error_rate(ground_truth_sort, LLR):
    """Вычислить EER и порог, где FNR и FPR наиболее близки."""
    thresholds, false_negative_rates, false_positive_rates = decision_curve(
        ground_truth_sort,
        LLR,
    )
    best_index = int(np.argmin(np.abs(false_negative_rates - false_positive_rates)))
    eer = 0.5 * (
        false_negative_rates[best_index] + false_positive_rates[best_index]
    )
    return float(eer * 100.0), float(thresholds[best_index])
