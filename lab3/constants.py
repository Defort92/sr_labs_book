"""Константы третьей лабораторной работы."""

from pathlib import Path


# Директории и файлы
LAB_DIR = Path(__file__).resolve().parent
PROJECT_DIR = LAB_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
DEFAULT_RESULTS_DIR = LAB_DIR / "results"
DEFAULT_CHECKPOINT_DIR = DATA_DIR / "lab3_models"
DEFAULT_AUGMENTED_CHECKPOINT_DIR = DATA_DIR / "lab3_models_aug"
DEFAULT_VOXCELEB_DEV_DIR = DATA_DIR / "voxceleb1_dev" / "wav"
DEFAULT_VOXCELEB_TEST_DIR = DATA_DIR / "voxceleb1_test" / "wav"
IDENTIFICATION_PROTOCOL_PATH = DATA_DIR / "voxceleb1_test" / "iden_split.txt"
DATASET_LIST_PATH = DATA_DIR / "lists" / "datasets.txt"
CONCATENATION_LIST_PATH = DATA_DIR / "lists" / "concat_arch.txt"
PROTOCOL_LIST_PATH = DATA_DIR / "lists" / "protocols.txt"
AUGMENTATION_DATASET_LIST_PATH = DATA_DIR / "lists" / "augment_datasets.txt"
MUSAN_PATH = DATA_DIR / "musan_split"
RIR_PATH = DATA_DIR / "RIRS_NOISES" / "simulated_rirs"
VOXCELEB_DEV_ARCHIVE_PATH = DATA_DIR / "vox1_dev_wav.zip"
VOXCELEB_DEV_EXTRACT_PATH = DATA_DIR / "voxceleb1_dev"
VOXCELEB_TEST_DOWNLOAD_PATH = DATA_DIR / "voxceleb1_test"
MUSAN_ARCHIVE_PATH = DATA_DIR / "musan.tar.gz"
RIR_ARCHIVE_PATH = DATA_DIR / "rirs_noises.zip"
RIR_EXTRACTION_TARGETS = (
    "RIRS_NOISES/simulated_rirs/mediumroom",
    "RIRS_NOISES/simulated_rirs/smallroom",
)

TRAINING_PLOT_FILENAME = "01_обучение_моделей.png"
COMPARISON_PLOT_FILENAME = "02_сравнение_точности.png"
SUMMARY_FILENAME = "сводка.json"
CHECKPOINT_FILENAME_TEMPLATE = "lab3_model_{epoch:04d}.pth"


# Параметры акустических признаков
SAMPLE_RATE = 16_000
MEL_FILTERS = 40
FFT_SIZE = 512
WINDOW_LENGTH = 400
HOP_LENGTH = 160
PREEMPHASIS_COEFFICIENT = 0.97
FEATURE_EPSILON = 1e-6
STATISTICS_EPSILON = 1e-5


# Архитектура и обучение полного эксперимента
RESNET_LAYERS = (3, 4, 6, 3)
RESNET_FILTERS = (32, 64, 128, 256)
EMBEDDING_SIZE = 512
ENCODER_STATISTICAL_POOLING = "SP"
ENCODER_ATTENTIVE_POOLING = "ASP"
ATTENTION_HIDDEN_SIZE = 128
ANGULAR_MARGIN = 0.35
ANGULAR_SCALE = 32.0
TRAIN_MAX_FRAMES = 200
EVAL_MAX_FRAMES = 1_000
TRAIN_BATCH_SIZE = 128
EVAL_BATCH_SIZE = 128
TRAIN_WORKERS = 5
EVAL_WORKERS = 5
TRAIN_SHUFFLE = True
PIN_MEMORY = False
LEARNING_RATE = 2.5
WEIGHT_DECAY = 0.0
VALIDATION_INTERVAL = 5
MAX_EPOCHS = 40
CHECKPOINT_RESUME_ENABLED = False
CHECKPOINT_RESUME_EPOCH = 4
ONE_CYCLE_START_FRACTION = 0.30
ONE_CYCLE_DIV_FACTOR = 20.0
ONE_CYCLE_FINAL_DIV_FACTOR = 10_000.0


# Режимы аугментации класса AugmentWAV
AUGMENTATION_NONE = 0
AUGMENTATION_REVERB = 1
AUGMENTATION_MUSIC = 2
AUGMENTATION_SPEECH = 3
AUGMENTATION_NOISE = 4
AUGMENTATION_MODE_COUNT = 5
NOISE_CATEGORY_MUSIC = "music"
NOISE_CATEGORY_SPEECH = "speech"
NOISE_CATEGORY_NOISE = "noise"


# Параметры быстрого демонстрационного эксперимента
DEMO_RANDOM_SEED = 42
DEMO_SPEAKER_COUNT = 3
DEMO_TRAIN_EXAMPLES_PER_SPEAKER = 15
DEMO_TEST_EXAMPLES_PER_SPEAKER = 6
DEMO_AUDIO_SECONDS = 0.45
DEMO_BASE_FREQUENCIES = (150.0, 260.0, 390.0)
DEMO_TRAIN_NOISE = 0.015
DEMO_AUGMENTED_TRAIN_NOISE = 0.08
DEMO_TEST_NOISE = 0.16
DEMO_EPOCHS = 7
DEMO_BATCH_SIZE = 6
DEMO_LEARNING_RATE = 0.03
DEMO_LAYERS = (1, 1, 1, 1)
DEMO_FILTERS = (4, 8, 16, 32)
DEMO_MEL_FILTERS = 16
DEMO_EMBEDDING_SIZE = 24
DEMO_MAX_THREADS = 1
PLOT_DPI = 150


# Технические режимы планировщика
SCHEDULER_STEP_EPOCH = "epoch"
SCHEDULER_STEP_ITERATION = "iteration"


# Переменные окружения для доступа к VoxCeleb
VOXCELEB_USER_ENV = "VOXCELEB_USER"
VOXCELEB_PASSWORD_ENV = "VOXCELEB_PASSWORD"
TEXT_ENCODING = "utf-8"


# Русские сообщения
MODEL_INITIALIZATION_MESSAGE = (
    "Размер дикторского эмбеддинга: {embedding_size}; "
    "тип статистического пулинга: {encoder_type}."
)
SOFTMAX_INITIALIZATION_MESSAGE = "Функция потерь Softmax инициализирована."
AM_SOFTMAX_INITIALIZATION_MESSAGE = (
    "Функция потерь AM-Softmax инициализирована: отступ={margin:.3f}, "
    "масштаб={scale:.3f}."
)
AAM_SOFTMAX_INITIALIZATION_MESSAGE = (
    "Функция потерь AAM-Softmax инициализирована: отступ={margin:.3f}, "
    "масштаб={scale:.3f}."
)
ADAM_INITIALIZATION_MESSAGE = "Оптимизатор Adam инициализирован."
SGD_INITIALIZATION_MESSAGE = "Оптимизатор SGD инициализирован."
STEP_SCHEDULER_INITIALIZATION_MESSAGE = "Планировщик StepLR инициализирован."
ONE_CYCLE_INITIALIZATION_MESSAGE = "Планировщик OneCycleLR инициализирован."
TRAIN_BATCH_MESSAGE = (
    "Эпоха {epoch:.0f}, пакет {batch:.0f}, скорость обучения {learning_rate:.6f}, "
    "потери {loss:.6f}, точность {accuracy:.3f}%"
)
EPOCH_RESULT_MESSAGE = (
    "Эпоха {epoch:.0f}: потери на обучении {loss:.6f}, "
    "точность на обучении {accuracy:.3f}%"
)
VALIDATION_RESULT_MESSAGE = (
    "Эпоха {epoch:.0f}: точность на валидации {accuracy:.3f}%"
)
TEST_RESULT_MESSAGE = "Точность на тестовой выборке: {accuracy:.3f}%"
DEMO_TEST_RESULT_MESSAGE = (
    "{model_name}, {test_name}: точность на тестовой выборке {accuracy:.3f}%"
)
FULL_TEST_RESULT_MESSAGE = "{model_name}: точность на тестовой выборке {accuracy:.3f}%"
DEMO_COMPLETE_MESSAGE = "Демонстрационный запуск завершён. Результаты: {path}"
LAB_COMPLETE_MESSAGE = "Лабораторная работа выполнена. Результаты: {path}"


# Сообщения об ошибках
ERROR_INVALID_TRAINING_LINE = "Некорректная строка списка данных: {line}"
ERROR_AUGMENTATION_MODE = "Неизвестный режим аугментации: {mode}"
ERROR_UNKNOWN_ENCODER = "Неизвестный тип статистического пулинга: {encoder_type}"
ERROR_INPUT_DIMENSIONS = "Входной аудиотензор должен иметь две размерности"
ERROR_EMPTY_LOADER = "Загрузчик данных не содержит ни одного пакета"
ERROR_SCHEDULER_MODE = "Неизвестный режим работы планировщика: {mode}"
ERROR_VOXCELEB_CREDENTIALS = (
    "Для загрузки VoxCeleb задайте переменные окружения VOXCELEB_USER "
    "и VOXCELEB_PASSWORD"
)
ERROR_DATASET_MISSING = (
    "Данные VoxCeleb1 не найдены. Запустите режим --demo или загрузите набор "
    "данных. Отсутствует: {path}"
)
ERROR_PREEMPHASIS_DIMENSIONS = (
    "Входной тензор для предыскажения должен иметь две размерности"
)
ERROR_AUDIO_EMPTY = "Аудиозапись не должна быть пустой: {path}"
ERROR_NOISE_FILES_MISSING = "Не найдены файлы шума категории: {category}"
ERROR_RIR_FILES_MISSING = "Не найдены импульсные отклики помещений"


# Интерфейс командной строки
CLI_DESCRIPTION = (
    "Выполнение лабораторной работы №3: построение и сравнение "
    "нейросетевых дикторских моделей."
)
CLI_DEMO_HELP = "Запустить быстрый эксперимент на синтетических голосах"
CLI_OUTPUT_DIR_HELP = "Директория для графиков и файла сводки"
CLI_HELP_HELP = "Показать эту справку и завершить работу"
CLI_OPTIONS_TITLE = "параметры"
CLI_DIRECTORY_METAVAR = "ДИРЕКТОРИЯ"
CLI_USAGE_PREFIX = "usage:"
CLI_RUSSIAN_USAGE_PREFIX = "использование:"
CLI_OPTIONS_PREFIX = "options:"
CLI_RUSSIAN_OPTIONS_PREFIX = "параметры:"


# Подписи результатов и графиков
MODEL_WITHOUT_AUGMENTATION = "модель без аугментации"
MODEL_WITH_AUGMENTATION = "модель с аугментацией"
TEST_CLEAN = "чистые тестовые записи"
TEST_NOISY = "зашумлённые тестовые записи"
TITLE_TRAINING_HISTORY = "Обучение дикторских моделей"
TITLE_ACCURACY_COMPARISON = "Сравнение точности моделей"
LABEL_EPOCH = "Эпоха"
LABEL_LOSS = "Значение функции потерь"
LABEL_ACCURACY_PERCENT = "Точность, %"
LABEL_TRAIN_LOSS_WITHOUT_AUGMENTATION = "Потери без аугментации"
LABEL_TRAIN_LOSS_WITH_AUGMENTATION = "Потери с аугментацией"
SUMMARY_KEY_PARAMETERS = "параметры"
SUMMARY_KEY_RESULTS = "результаты"
SUMMARY_KEY_HISTORY = "история_обучения"
