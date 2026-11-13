"""Константы четвёртой лабораторной работы."""

from pathlib import Path


# Директории и файлы
LAB_DIR = Path(__file__).resolve().parent
PROJECT_DIR = LAB_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
DEFAULT_RESULTS_DIR = LAB_DIR / "results"
DEFAULT_VOXCELEB_DIR = DATA_DIR / "voxceleb1_test" / "wav"
DEFAULT_PROTOCOL_PATH = DATA_DIR / "voxceleb1_test" / "veri_test2.txt"
DEFAULT_MODEL_DIR = DATA_DIR / "models"
PRETRAINED_MODEL_FILENAME = "baseline_v2_ap.model"
PRETRAINED_MODEL_PATH = DEFAULT_MODEL_DIR / PRETRAINED_MODEL_FILENAME
DATASET_LIST_PATH = DATA_DIR / "lists" / "datasets.txt"
PROTOCOL_LIST_PATH = DATA_DIR / "lists" / "protocols.txt"
MODEL_LIST_PATH = DATA_DIR / "lists" / "models.txt"
VOXCELEB_ARCHIVE_PATH = DATA_DIR / "vox1_test_wav.zip"
VOXCELEB_EXTRACT_PATH = DATA_DIR / "voxceleb1_test"

HISTOGRAM_PLOT_FILENAME = "01_распределения_оценок.png"
LLR_PLOT_FILENAME = "02_отношение_правдоподобия.png"
ERRORS_PLOT_FILENAME = "03_ошибки_по_порогам.png"
SUMMARY_FILENAME = "сводка.json"


# Параметры модели и признаков
SAMPLE_RATE = 16_000
FFT_SIZE = 512
WINDOW_LENGTH = 400
HOP_LENGTH = 160
MEL_FILTERS = 64
EMBEDDING_SIZE = 512
RESNET_FILTERS = (32, 64, 128, 256)
RESNET_LAYERS = (3, 4, 6, 3)
ATTENTION_HIDDEN_SIZE = 128
ENCODER_SELF_ATTENTIVE_POOLING = "SAP"
ENCODER_ATTENTIVE_STATISTICS_POOLING = "ASP"
FEATURE_EPSILON = 1e-6
STATISTICS_EPSILON = 1e-5
PREEMPHASIS_COEFFICIENT = 0.97


# Параметры полного тестирования
EVALUATION_FRAMES = 200
EVALUATION_SEGMENTS = 5
EVALUATION_WORKERS = 0
TARGET_LABEL = 1
IMPOSTOR_LABEL = 0
NUMERICAL_EPSILON = 1e-12


# Параметры критериев принятия решения
DEFAULT_TARGET_PRIOR = 0.5
DEFAULT_NEYMAN_PEARSON_FNR = 0.25
DEFAULT_COST_CORRECT_TARGET = 0.0
DEFAULT_COST_MISSED_TARGET = 1.0
DEFAULT_COST_FALSE_ACCEPT = 1.0
DEFAULT_COST_CORRECT_IMPOSTOR = 0.0
MINIMAX_PRIOR_STEPS = 1_000


# Параметры демонстрации
DEMO_RANDOM_SEED = 42
DEMO_TARGET_COUNT = 700
DEMO_IMPOSTOR_COUNT = 1_400
DEMO_TARGET_MEAN = 0.72
DEMO_TARGET_STD = 0.13
DEMO_IMPOSTOR_MEAN = 0.24
DEMO_IMPOSTOR_STD = 0.16
PLOT_DPI = 150


# Доступ к VoxCeleb
VOXCELEB_USER_ENV = "VOXCELEB_USER"
VOXCELEB_PASSWORD_ENV = "VOXCELEB_PASSWORD"
TEXT_ENCODING = "utf-8"


# Русские сообщения модели и загрузки
MODEL_INITIALIZATION_MESSAGE = (
    "Размер дикторского эмбеддинга: {embedding_size}; "
    "тип статистического пулинга: {encoder_type}."
)
MODEL_DOWNLOADED_MESSAGE = "Файл модели загружен: {filename}"
MODEL_LOADED_MESSAGE = "Предобученные параметры модели загружены."
SCORING_PROGRESS_DESCRIPTION = "Вычисление оценок"


# Сообщения об ошибках
ERROR_SCORES_ONE_DIMENSIONAL = "Оценки и метки должны быть одномерными массивами"
ERROR_SCORES_LENGTH = "Количество оценок должно совпадать с количеством меток"
ERROR_SCORES_EMPTY = "Массив оценок не должен быть пустым"
ERROR_CLASS_EMPTY = "В протоколе должны присутствовать таргет- и импостор-пары"
ERROR_LABEL_VALUE = "Метки должны принимать только значения 0 и 1"
ERROR_PRIOR_RANGE = "Априорная вероятность должна находиться в диапазоне от 0 до 1"
ERROR_FNR_RANGE = "Заданная вероятность пропуска должна находиться от 0 до 1"
ERROR_COSTS = "Стоимости ошибок должны превышать стоимости правильных решений"
ERROR_UNKNOWN_ENCODER = "Неизвестный тип статистического пулинга: {encoder_type}"
ERROR_PREEMPHASIS_DIMENSIONS = (
    "Входной тензор для предыскажения должен иметь две размерности"
)
ERROR_MODEL_DOWNLOAD = (
    "Не удалось загрузить модель по адресу {url}. "
    "Проверьте соединение или поместите файл модели вручную."
)
ERROR_MODEL_WEIGHTS = "В файле не найдены параметры генератора эмбеддингов"
ERROR_VOXCELEB_CREDENTIALS = (
    "Для загрузки VoxCeleb задайте переменные окружения VOXCELEB_USER "
    "и VOXCELEB_PASSWORD"
)
ERROR_DATA_MISSING = (
    "Не найдены данные полного эксперимента. Запустите режим --demo или "
    "подготовьте файл: {path}"
)


# Интерфейс командной строки
CLI_DESCRIPTION = (
    "Выполнение лабораторной работы №4: критерии принятия решения "
    "и метрики качества верификации диктора."
)
CLI_DEMO_HELP = "Использовать синтетические оценки сравнения"
CLI_OUTPUT_DIR_HELP = "Директория для графиков и файла сводки"
CLI_HELP_HELP = "Показать эту справку и завершить работу"
CLI_OPTIONS_TITLE = "параметры"
CLI_DIRECTORY_METAVAR = "ДИРЕКТОРИЯ"
CLI_USAGE_PREFIX = "usage:"
CLI_RUSSIAN_USAGE_PREFIX = "использование:"
CLI_OPTIONS_PREFIX = "options:"
CLI_RUSSIAN_OPTIONS_PREFIX = "параметры:"


# Подписи графиков
TITLE_SCORE_DISTRIBUTIONS = "Распределения оценок сравнения"
TITLE_LLR = "Логарифм отношения правдоподобия"
TITLE_THRESHOLD_ERRORS = "Зависимость ошибок от порога"
TITLE_ERROR_PROBABILITY = "Вероятность ошибки принятия решения"
LABEL_SCORE = "Оценка сходства"
LABEL_DENSITY = "Плотность распределения"
LABEL_LLR = "LLR"
LABEL_THRESHOLD = "Порог LLR"
LABEL_ERROR_RATE = "Вероятность ошибки"
LABEL_ERROR_PROBABILITY = "Вероятность ошибки Pₑ"
LABEL_TARGET_SCORES = "Таргет-оценки"
LABEL_IMPOSTOR_SCORES = "Импостор-оценки"
LABEL_TARGET_DENSITY = "Плотность таргет-оценок"
LABEL_IMPOSTOR_DENSITY = "Плотность импостор-оценок"
LABEL_FNR = "FNR — пропуск своего диктора"
LABEL_FPR = "FPR — допуск чужого диктора"


# Русские названия результатов
CRITERION_MAP = "максимум апостериорной вероятности"
CRITERION_NEYMAN_PEARSON = "Нейман–Пирсон"
CRITERION_BAYES = "Байес"
CRITERION_MINIMAX = "минимакс"
RESULT_MESSAGE = (
    "{criterion}: порог={threshold:.4f}, FNR={fnr:.4f}, "
    "FPR={fpr:.4f}, риск={risk:.4f}"
)
NEYMANN_PEARSON_RESULT_MESSAGE = (
    "{criterion}: порог={threshold:.4f}, заданный FNR={fnr:.4f}, FPR={fpr:.4f}"
)
MINIMAX_RESULT_MESSAGE = (
    "{criterion}: порог={threshold:.4f}, FNR={fnr:.4f}, FPR={fpr:.4f}, "
    "риск={risk:.4f}, наименее благоприятная вероятность H₀={prior:.4f}"
)
EER_RESULT_MESSAGE = "Равновероятная ошибка EER: {eer:.3f}%"
DEMO_COMPLETE_MESSAGE = "Демонстрационный запуск завершён. Результаты: {path}"
LAB_COMPLETE_MESSAGE = "Лабораторная работа выполнена. Результаты: {path}"


# Ключи JSON-сводки
SUMMARY_KEY_MODE = "режим"
SUMMARY_KEY_PARAMETERS = "параметры"
SUMMARY_KEY_RESULTS = "результаты"
SUMMARY_KEY_THRESHOLD = "порог"
SUMMARY_KEY_FNR = "fnr"
SUMMARY_KEY_FPR = "fpr"
SUMMARY_KEY_RISK = "риск"
SUMMARY_KEY_PRIOR = "вероятность_целевой_гипотезы"
SUMMARY_KEY_EER = "eer_проценты"
SUMMARY_KEY_EER_THRESHOLD = "порог_eer"
SUMMARY_KEY_TARGET_PRIOR = "априорная_вероятность_H0"
SUMMARY_KEY_FNR_LIMIT = "ограничение_fnr"
SUMMARY_KEY_TARGET_COUNT = "число_таргет_сравнений"
SUMMARY_KEY_IMPOSTOR_COUNT = "число_импостор_сравнений"
MODE_DEMO = "синтетические оценки сравнения"
MODE_VOXCELEB = "реальные данные VoxCeleb1-O cleaned"
