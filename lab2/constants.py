"""Константы второй лабораторной работы."""

from pathlib import Path


# Директории и файлы
LAB_DIR = Path(__file__).resolve().parent
PROJECT_DIR = LAB_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
DEFAULT_RESULTS_DIR = LAB_DIR / "results"
DEFAULT_RTTM_DIR = LAB_DIR / "ground_truth" / "rttm"
DEFAULT_VOXCELEB_DIR = DATA_DIR / "voxceleb1_test" / "wav"
IMPULSE_RESPONSE_PATH = LAB_DIR / "cathIR.wav"
DATASET_LIST_PATH = DATA_DIR / "lists" / "datasets.txt"
VOXCELEB_ARCHIVE_PATH = DATA_DIR / "vox1_test_wav.zip"
VOXCELEB_EXTRACT_PATH = DATA_DIR / "voxceleb1_test"

EXAMPLE_RTTM_FILENAME = "id10271_1gtz-CUIygI_00006.rttm"
EXAMPLE_AUDIO_RELATIVE_PATH = Path("id10271") / "1gtz-CUIygI" / "00006.wav"

VAD_PLOT_FILENAME = "01_разметка_vad.png"
ENERGY_PLOT_FILENAME = "02_распределение_энергии.png"
METRICS_PLOT_FILENAME = "03_метрики_искажений.png"
SUMMARY_FILENAME = "сводка.json"


# Параметры сигнала и VAD
TARGET_SAMPLE_RATE = 16_000
FRAME_WINDOW = 320
FRAME_SHIFT = 160
GMM_COMPONENTS = 3
GMM_REALIGNMENTS = 10
GMM_INITIAL_WEIGHTS = (0.33, 0.33, 0.33)
GMM_INITIAL_MEANS = (-1.0, 0.0, 1.0)
GMM_INITIAL_SIGMAS = (1.0, 1.0, 1.0)
VAD_THRESHOLD = 0.3
MORPHOLOGY_MASK_SIZE = 6_000
NUMERICAL_EPSILON = 1e-10


# Параметры демонстрационного сигнала
DEMO_RANDOM_SEED = 42
DEMO_DURATION_SECONDS = 10.0
DEMO_SPEECH_INTERVALS = (
    (0.8, 2.4),
    (3.0, 4.7),
    (5.2, 7.4),
    (8.0, 9.4),
)
DEMO_SPEECH_FREQUENCIES = (125.0, 180.0, 240.0)
DEMO_SILENCE_NOISE_LEVEL = 0.002
DEMO_SPEECH_NOISE_LEVEL = 0.01
AWGN_LEVELS = (0.10, 0.30, 0.50)
NOTEBOOK_AWGN_SIGMA = 0.15
PLOT_DPI = 150


# Значения форматов и режимов
RTTM_RECORD_TYPE = "SPEAKER"
AUGMENTATION_MODE_REVERB = "reverb"
AUGMENTATION_MODE_AWGN = "awgn"
AUGMENTATION_CLEAN = "чистый сигнал"
AUGMENTATION_REVERB = "реверберация"
AUGMENTATION_AWGN_TEMPLATE = "АБГШ, σ={sigma:.2f}"
MODE_SYNTHETIC_DEMO = "демонстрация на синтетическом сигнале"
MODE_VOXCELEB = "эксперимент на VoxCeleb1 test"
TEXT_ENCODING = "utf-8"
VOXCELEB_USER_ENV = "VOXCELEB_USER"
VOXCELEB_PASSWORD_ENV = "VOXCELEB_PASSWORD"


# Сообщения об ошибках
ERROR_SIGNAL_ONE_DIMENSIONAL = "Сигнал должен быть одномерным"
ERROR_FRAMES_TWO_DIMENSIONAL = "Матрица фреймов должна быть двумерной"
ERROR_SIGNAL_EMPTY = "Сигнал не должен быть пустым"
ERROR_ENERGY_EMPTY = "Массив энергий не должен быть пустым"
ERROR_WINDOW_AND_SHIFT_POSITIVE = "Размер и шаг окна должны быть положительными"
ERROR_GMM_REALIGNMENTS_POSITIVE = "Количество итераций EM должно быть положительным"
ERROR_GMM_DIMENSIONS = "Размерности параметров GMM не совпадают"
ERROR_NOISE_SIGMA = "Стандартное отклонение шума не может быть отрицательным"
ERROR_IMPULSE_RESPONSE_EMPTY = "Импульсный отклик не должен быть пустым"
ERROR_MARKUP_LENGTHS = "Эталонная и системная разметки должны иметь одинаковую длину"
ERROR_RTTM_FORMAT = "Некорректная строка RTTM: {line}"
ERROR_NO_RTTM_FILES = "В директории не найдены RTTM-файлы: {path}"
ERROR_AUDIO_MISSING = (
    "Не найдены аудиофайлы VoxCeleb1 test. Запустите демонстрационный режим "
    "с параметром --demo или загрузите набор данных.\n"
    "Примеры отсутствующих файлов:\n{examples}"
)
ERROR_VOXCELEB_CREDENTIALS = (
    "Для загрузки VoxCeleb задайте переменные окружения VOXCELEB_USER "
    "и VOXCELEB_PASSWORD"
)


# Русский интерфейс командной строки
CLI_DESCRIPTION = (
    "Выполнение лабораторной работы №2: энергетический детектор речевой "
    "активности на основе модели гауссовых смесей."
)
CLI_DEMO_HELP = "Использовать синтетический сигнал вместо записей VoxCeleb"
CLI_OUTPUT_DIR_HELP = "Директория для графиков и файла сводки"
CLI_VOXCELEB_DIR_HELP = "Корневая директория аудиофайлов VoxCeleb1 test"
CLI_RTTM_DIR_HELP = "Директория с эталонными RTTM-файлами"
CLI_HELP_HELP = "Показать эту справку и завершить работу"
CLI_OPTIONS_TITLE = "параметры"
CLI_DIRECTORY_METAVAR = "ДИРЕКТОРИЯ"
CLI_USAGE_PREFIX = "usage:"
CLI_RUSSIAN_USAGE_PREFIX = "использование:"
CLI_OPTIONS_PREFIX = "options:"
CLI_RUSSIAN_OPTIONS_PREFIX = "параметры:"
DEMO_COMPLETE_MESSAGE = "Демонстрационный запуск завершён. Результаты: {path}"
LAB_COMPLETE_MESSAGE = "Лабораторная работа выполнена. Результаты: {path}"


# Подписи графиков
TITLE_VAD_MARKUP = "Сигнал и разметка детектора речевой активности"
TITLE_ENERGY_HISTOGRAM = "Распределение нормализованной энергии фреймов"
TITLE_GMM_DENSITY = "Плотность распределения энергии по модели GMM"
TITLE_DISTORTION_METRICS = "Влияние искажений на качество VAD"
LABEL_SAMPLE_NUMBER = "Номер отсчёта, n"
LABEL_TIME_SECONDS = "Время, с"
LABEL_AMPLITUDE = "Амплитуда"
LABEL_NORMALIZED_ENERGY = "Нормализованная энергия"
LABEL_DENSITY = "Плотность распределения"
LABEL_METRIC_VALUE = "Значение метрики"
LABEL_IDEAL_VAD = "Эталонная разметка"
LABEL_RAW_VAD = "Разметка до морфологической обработки"
LABEL_FILTERED_VAD = "Итоговая разметка"
LABEL_SIGNAL = "Сигнал"
LABEL_FNR = "FNR — пропуск речи"
LABEL_FPR = "FPR — ложное обнаружение"
LABEL_PRECISION = "Точность"
LABEL_RECALL = "Полнота"
LABEL_GMM_MIXTURE = "Смесь GMM"
GMM_COMPONENT_LABEL_TEMPLATE = "Компонент GMM {index}"


# Сообщения и ключи сводки
METRICS_MESSAGE = (
    "{name}: FNR={fnr:.3f}, FPR={fpr:.3f}, "
    "точность={precision:.3f}, полнота={recall:.3f}"
)
SUMMARY_KEY_MODE = "режим"
SUMMARY_KEY_PARAMETERS = "параметры_vad"
SUMMARY_KEY_WINDOW = "размер_окна"
SUMMARY_KEY_SHIFT = "шаг_окна"
SUMMARY_KEY_REALIGNMENTS = "итерации_em"
SUMMARY_KEY_THRESHOLD = "порог_vad"
SUMMARY_KEY_MORPHOLOGY = "размер_морфологической_маски"
SUMMARY_KEY_RESULTS = "результаты"
SUMMARY_KEY_FNR = "fnr"
SUMMARY_KEY_FPR = "fpr"
SUMMARY_KEY_PRECISION = "точность"
SUMMARY_KEY_RECALL = "полнота"
SUMMARY_KEY_TP = "истинно_положительные"
SUMMARY_KEY_FP = "ложно_положительные"
SUMMARY_KEY_FN = "ложно_отрицательные"
SUMMARY_KEY_TN = "истинно_отрицательные"
