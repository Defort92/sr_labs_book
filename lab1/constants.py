"""Константы первой лабораторной работы."""

from pathlib import Path


# Настройки файлов и директорий
LAB_DIR = Path(__file__).resolve().parent
DEFAULT_METADATA_PATH = LAB_DIR / "metadata" / "meta.txt"
DEFAULT_RESULTS_DIR = LAB_DIR / "results"

PREEMPHASIS_PLOT_FILENAME = "01_преэмфазис.png"
MEL_FILTER_BANK_PLOT_FILENAME = "02_мел_банк_фильтров.png"
FEATURE_MAPS_PLOT_FILENAME = "03_карты_признаков.png"
GENDER_LOG_MEL_PLOT_FILENAME = "04_распределения_log_mel.png"
GENDER_MFCC_PLOT_FILENAME = "05_распределения_mfcc.png"
SUMMARY_FILENAME = "сводка.json"


# Настройки обработки сигнала
TARGET_SAMPLE_RATE = 16_000
FRAME_SIZE_SECONDS = 0.025
FRAME_STRIDE_SECONDS = 0.01
PREEMPHASIS_COEFFICIENT = 0.97
NFFT = 512
N_FILTERS = 40
N_MFCC = 20
MVN_LEFT_CONTEXT = 150
MVN_RIGHT_CONTEXT = 150
ANALYSIS_DURATION_SECONDS = 3.5
FRAME_SIZE_MILLISECONDS = 25
FRAME_STRIDE_MILLISECONDS = 10


# Настройки демонстрации и графиков
DEMO_RANDOM_SEED = 42
DEMO_RECORDS_PER_GENDER = 3
DEMO_MALE_BASE_FREQUENCY = 115.0
DEMO_FEMALE_BASE_FREQUENCY = 205.0
DEMO_FREQUENCY_STEP = 7.0
DEMO_ENVELOPE_FREQUENCY = 2.3
DEMO_NOISE_LEVEL = 0.02
DEMO_HARMONICS = 8
PLOT_DPI = 150
SPECTROGRAM_NFFT = 1024
SPECTROGRAM_OVERLAP = 900
HISTOGRAM_MIN_BINS = 20
HISTOGRAM_MAX_BINS = 150


# Значения из формата метаданных
MALE_GENDER_CODE = "m"
FEMALE_GENDER_CODE = "f"
SUPPORTED_GENDER_CODES = {MALE_GENDER_CODE, FEMALE_GENDER_CODE}


# Сообщения об ошибках
ERROR_LINE_MUST_BE_STRING = "Строка метаданных должна иметь тип str"
ERROR_METADATA_LINE_FORMAT = (
    "Строка метаданных должна содержать идентификатор диктора, пол и путь к файлу"
)
ERROR_GENDER_CODE = "Код пола должен быть равен 'm' или 'f'"
ERROR_SIGNAL_ONE_DIMENSIONAL = "Сигнал должен быть одномерным"
ERROR_EMPHASIZED_SIGNAL_ONE_DIMENSIONAL = (
    "Сигнал после преэмфазиса должен быть одномерным"
)
ERROR_FRAME_PARAMETERS_POSITIVE = (
    "Частота дискретизации, размер и шаг фрейма должны быть положительными"
)
ERROR_FRAMES_TWO_DIMENSIONAL = "Матрица фреймов должна быть двумерной"
ERROR_NFFT_POSITIVE = "Размер БПФ должен быть положительным"
ERROR_FBANK_PARAMETERS_POSITIVE = (
    "Количество фильтров, частота дискретизации и размер БПФ должны быть положительными"
)
ERROR_SPECTRUM_AND_FILTERS_TWO_DIMENSIONAL = (
    "Спектр мощности и банк фильтров должны быть двумерными матрицами"
)
ERROR_SPECTRUM_FILTER_DIMENSIONS = (
    "Размерности спектра мощности и банка фильтров не совпадают"
)
ERROR_FILTER_BANK_FEATURES_TWO_DIMENSIONAL = (
    "Матрица признаков банка фильтров должна быть двумерной"
)
ERROR_MFCC_COUNT = (
    "Количество MFCC должно быть положительным и не превышать количество Mel-фильтров"
)
ERROR_AUDIO_DIMENSION = (
    "Аудиосигнал должен быть одноканальным или иметь форму [отсчёты, каналы]"
)


# Текст интерфейса командной строки
CLI_DESCRIPTION = (
    "Выполнение лабораторной работы №1 и сохранение полученных графиков. "
    "Полный режим использует записи VoxCeleb1 из metadata/meta.txt. "
    "Режим --demo проверяет конвейер на сгенерированных сигналах."
)
CLI_METADATA_HELP = "Путь к файлу метаданных VoxCeleb"
CLI_OUTPUT_DIR_HELP = "Директория для графиков и файла сводки"
CLI_DEMO_HELP = "Использовать сгенерированные сигналы вместо записей VoxCeleb"
CLI_HELP_HELP = "Показать эту справку и завершить работу"
CLI_OPTIONS_TITLE = "параметры"
CLI_PATH_METAVAR = "ПУТЬ"
CLI_DIRECTORY_METAVAR = "ДИРЕКТОРИЯ"
CLI_USAGE_PREFIX = "usage:"
CLI_RUSSIAN_USAGE_PREFIX = "использование:"
CLI_OPTIONS_PREFIX = "options:"
CLI_RUSSIAN_OPTIONS_PREFIX = "параметры:"
DEMO_COMPLETE_MESSAGE = "Демонстрационный запуск завершён. Результаты: {path}"
LAB_COMPLETE_MESSAGE = "Лабораторная работа выполнена. Результаты: {path}"
MISSING_AUDIO_MESSAGE = (
    "Аудиофайлы VoxCeleb не найдены. Скачайте и распакуйте VoxCeleb1 test "
    "с помощью ноутбука или запустите сценарий с параметром --demo.\n"
    "Примеры отсутствующих файлов:\n{examples}"
)
NOTEBOOK_LOADING_MESSAGE = "Загрузка файла: {path}"
NOTEBOOK_LOADING_COMPLETE_MESSAGE = "Загрузка завершена"
NOTEBOOK_FRAMES_SHAPE_MESSAGE = "Размер матрицы фреймов: {shape}"
NOTEBOOK_MALE_LOG_MEL_SHAPE_MESSAGE = (
    "Размер Log-Mel признаков мужских голосов: {shape}"
)
NOTEBOOK_FEMALE_LOG_MEL_SHAPE_MESSAGE = (
    "Размер Log-Mel признаков женских голосов: {shape}"
)
NOTEBOOK_MALE_MFCC_SHAPE_MESSAGE = "Размер MFCC мужских голосов: {shape}"
NOTEBOOK_FEMALE_MFCC_SHAPE_MESSAGE = "Размер MFCC женских голосов: {shape}"


# Подписи графиков
TITLE_ORIGINAL_WAVEFORM = "Исходная осциллограмма"
TITLE_ORIGINAL_SIGNAL = "Исходный сигнал"
TITLE_AFTER_PREEMPHASIS = "Сигнал после преэмфазиса"
TITLE_MEL_FILTER_BANK = "Mel-банк фильтров"
TITLE_LOG_MEL = "Логарифмы энергий Mel-фильтров"
TITLE_MFCC = "Мел-частотные кепстральные коэффициенты"
TITLE_NORMALIZED_LOG_MEL = "Нормализованные Log-Mel признаки"
TITLE_NORMALIZED_MFCC = "Нормализованные MFCC"
TITLE_GENDER_DISTRIBUTIONS = "Распределения мужских и женских голосов"

LABEL_TIME_SECONDS = "Время, с"
LABEL_AMPLITUDE = "Амплитуда"
LABEL_FREQUENCY_HZ = "Частота, Гц"
LABEL_FILTER_WEIGHT = "Вес фильтра"
LABEL_FRAME = "Номер фрейма"
LABEL_COMPONENT = "Компонента"
LABEL_FEATURE_VALUE = "Значение признака"
LABEL_DENSITY = "Плотность распределения"
LABEL_MALE = "мужские голоса"
LABEL_FEMALE = "женские голоса"
FEATURE_NAME_LOG_MEL = "Log-Mel"
FEATURE_NAME_MFCC = "MFCC"


# Значения сводного отчёта
MODE_SYNTHETIC_DEMO = "демонстрация на синтетических сигналах"
MODE_VOXCELEB = "эксперимент на VoxCeleb1 test"

SUMMARY_KEY_MODE = "режим"
SUMMARY_KEY_RECORDINGS = "количество_записей"
SUMMARY_KEY_MALE_RECORDINGS = "количество_мужских_записей"
SUMMARY_KEY_FEMALE_RECORDINGS = "количество_женских_записей"
SUMMARY_KEY_LOG_MEL_SHAPE = "размер_log_mel_первой_записи"
SUMMARY_KEY_MFCC_SHAPE = "размер_mfcc_первой_записи"
SUMMARY_KEY_PARAMETERS = "параметры"
SUMMARY_KEY_SAMPLE_RATE = "частота_дискретизации"
SUMMARY_KEY_FRAME_MS = "размер_фрейма_мс"
SUMMARY_KEY_FRAME_SHIFT_MS = "шаг_фрейма_мс"
SUMMARY_KEY_NFFT = "размер_бпф"
SUMMARY_KEY_MEL_FILTERS = "количество_mel_фильтров"
SUMMARY_KEY_MFCC = "количество_mfcc"
