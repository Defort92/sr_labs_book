"""Основные упражнения третьей лабораторной работы."""

from __future__ import annotations

import os
import random

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset

from constants import (
    ATTENTION_HIDDEN_SIZE,
    AUGMENTATION_MODE_COUNT,
    AUGMENTATION_MUSIC,
    AUGMENTATION_NOISE,
    AUGMENTATION_NONE,
    AUGMENTATION_REVERB,
    AUGMENTATION_SPEECH,
    ENCODER_ATTENTIVE_POOLING,
    ENCODER_STATISTICAL_POOLING,
    ERROR_AUGMENTATION_MODE,
    ERROR_EMPTY_LOADER,
    ERROR_INVALID_TRAINING_LINE,
    ERROR_SCHEDULER_MODE,
    ERROR_UNKNOWN_ENCODER,
    FEATURE_EPSILON,
    FFT_SIZE,
    HOP_LENGTH,
    MODEL_INITIALIZATION_MESSAGE,
    NOISE_CATEGORY_MUSIC,
    NOISE_CATEGORY_NOISE,
    NOISE_CATEGORY_SPEECH,
    SAMPLE_RATE,
    SCHEDULER_STEP_EPOCH,
    SCHEDULER_STEP_ITERATION,
    STATISTICS_EPSILON,
    TRAIN_BATCH_MESSAGE,
    WINDOW_LENGTH,
)
from audio_data import AugmentWAV, loadWAV
from preproc import PreEmphasis


def _parse_dataset_list(data_list, data_path):
    """Преобразовать список «диктор файл» в пути и числовые метки."""
    parsed_lines = []
    for raw_line in data_list:
        fields = raw_line.strip().split()
        if len(fields) != 2:
            raise ValueError(
                ERROR_INVALID_TRAINING_LINE.format(line=raw_line.strip())
            )
        parsed_lines.append(fields)

    speaker_ids = sorted({fields[0] for fields in parsed_lines})
    speaker_to_label = {
        speaker_id: index for index, speaker_id in enumerate(speaker_ids)
    }
    file_paths = [os.path.join(data_path, fields[1]) for fields in parsed_lines]
    labels = [speaker_to_label[fields[0]] for fields in parsed_lines]
    return file_paths, labels


class train_dataset_loader(Dataset):
    """Загрузчик тренировочных записей с необязательной онлайн-аугментацией."""

    def __init__(
        self,
        train_list,
        max_frames,
        train_path,
        augment=False,
        musan_path=None,
        rir_path=None,
    ):
        self.max_frames = max_frames
        self.augment = augment
        if self.augment:
            self.augment_wav = AugmentWAV(
                musan_path=musan_path,
                rir_path=rir_path,
                max_frames=max_frames,
            )

        self.data_list, self.data_label = _parse_dataset_list(
            train_list,
            train_path,
        )

    def __getitem__(self, index):
        audio = loadWAV(
            self.data_list[index],
            self.max_frames,
            evalmode=False,
        )

        if self.augment:
            augmentation_mode = random.randrange(AUGMENTATION_MODE_COUNT)
            if augmentation_mode == AUGMENTATION_REVERB:
                audio = self.augment_wav.reverberate(audio)
            elif augmentation_mode == AUGMENTATION_MUSIC:
                audio = self.augment_wav.additive_noise(
                    NOISE_CATEGORY_MUSIC,
                    audio,
                )
            elif augmentation_mode == AUGMENTATION_SPEECH:
                audio = self.augment_wav.additive_noise(
                    NOISE_CATEGORY_SPEECH,
                    audio,
                )
            elif augmentation_mode == AUGMENTATION_NOISE:
                audio = self.augment_wav.additive_noise(
                    NOISE_CATEGORY_NOISE,
                    audio,
                )
            elif augmentation_mode != AUGMENTATION_NONE:
                raise RuntimeError(
                    ERROR_AUGMENTATION_MODE.format(mode=augmentation_mode)
                )

        return torch.as_tensor(audio, dtype=torch.float32), self.data_label[index]

    def __len__(self):
        return len(self.data_list)


class test_dataset_loader(Dataset):
    """Загрузчик валидационных и тестовых записей."""

    def __init__(self, test_list, max_frames, test_path, num_eval=1):
        self.max_frames = max_frames
        self.num_eval = num_eval
        self.data_list, self.data_label = _parse_dataset_list(
            test_list,
            test_path,
        )

    def __getitem__(self, index):
        audio = loadWAV(
            self.data_list[index],
            self.max_frames,
            evalmode=True,
            num_eval=self.num_eval,
        )
        return torch.as_tensor(audio, dtype=torch.float32), self.data_label[index]

    def __len__(self):
        return len(self.data_list)


class MaxoutLinear(nn.Module):
    """Полносвязный Maxout-слой из двух параллельных преобразований."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.linear1 = nn.Linear(*args, **kwargs)
        self.linear2 = nn.Linear(*args, **kwargs)

    def forward(self, features):
        return torch.maximum(self.linear1(features), self.linear2(features))


class ResNet(nn.Module):
    """ResNet-модель, преобразующая аудиозапись в дикторский эмбеддинг."""

    def __init__(
        self,
        block,
        layers,
        activation,
        num_filters,
        nOut,
        encoder_type=ENCODER_STATISTICAL_POOLING,
        n_mels=64,
        log_input=True,
        **kwargs,
    ):
        super().__init__()
        print(
            MODEL_INITIALIZATION_MESSAGE.format(
                embedding_size=nOut,
                encoder_type=encoder_type,
            )
        )

        if encoder_type not in {
            ENCODER_STATISTICAL_POOLING,
            ENCODER_ATTENTIVE_POOLING,
        }:
            raise ValueError(
                ERROR_UNKNOWN_ENCODER.format(encoder_type=encoder_type)
            )

        self.inplanes = num_filters[0]
        self.encoder_type = encoder_type
        self.n_mels = n_mels
        self.log_input = log_input

        self.torchfb = nn.Sequential(
            PreEmphasis(),
            torchaudio.transforms.MelSpectrogram(
                sample_rate=SAMPLE_RATE,
                n_fft=FFT_SIZE,
                win_length=WINDOW_LENGTH,
                hop_length=HOP_LENGTH,
                window_fn=torch.hamming_window,
                n_mels=n_mels,
            ),
        )
        self.instancenorm = nn.InstanceNorm1d(n_mels)

        self.conv1 = nn.Conv2d(
            1,
            num_filters[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.relu = activation(inplace=True)

        self.layer1 = self._make_layer(
            block,
            num_filters[0],
            layers[0],
            stride=1,
            activation=activation,
        )
        self.layer2 = self._make_layer(
            block,
            num_filters[1],
            layers[1],
            stride=2,
            activation=activation,
        )
        self.layer3 = self._make_layer(
            block,
            num_filters[2],
            layers[2],
            stride=2,
            activation=activation,
        )
        self.layer4 = self._make_layer(
            block,
            num_filters[3],
            layers[3],
            stride=2,
            activation=activation,
        )

        frequency_bins = (self.n_mels + 7) // 8
        frame_feature_size = num_filters[3] * block.expansion * frequency_bins
        self.attention = nn.Sequential(
            nn.Conv1d(frame_feature_size, ATTENTION_HIDDEN_SIZE, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(ATTENTION_HIDDEN_SIZE),
            nn.Conv1d(ATTENTION_HIDDEN_SIZE, frame_feature_size, kernel_size=1),
            nn.Softmax(dim=2),
        )
        pooled_size = frame_feature_size * 2
        self.fc = nn.Sequential(
            MaxoutLinear(pooled_size, nOut),
            nn.BatchNorm1d(nOut, affine=False),
        )

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(
        self,
        block,
        planes,
        blocks,
        stride=1,
        activation=nn.ReLU,
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                activation=activation,
            )
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, activation=activation)
            )
        return nn.Sequential(*layers)

    def forward(self, audio):
        with torch.no_grad():
            features = self.torchfb(audio.float()) + FEATURE_EPSILON
            if self.log_input:
                features = features.log()
            features = self.instancenorm(features).unsqueeze(1)

        features = self.relu(self.bn1(self.conv1(features)))
        features = self.layer1(features)
        features = self.layer2(features)
        features = self.layer3(features)
        features = self.layer4(features)

        batch_size, channels, frequency_bins, time_frames = features.shape
        features = features.reshape(
            batch_size,
            channels * frequency_bins,
            time_frames,
        )

        if self.encoder_type == ENCODER_STATISTICAL_POOLING:
            mean = features.mean(dim=2)
            standard_deviation = features.std(dim=2, unbiased=False)
        else:
            attention_weights = self.attention(features)
            mean = torch.sum(features * attention_weights, dim=2)
            second_moment = torch.sum(
                features.square() * attention_weights,
                dim=2,
            )
            variance = (second_moment - mean.square()).clamp_min(
                STATISTICS_EPSILON
            )
            standard_deviation = torch.sqrt(variance)

        pooled_features = torch.cat((mean, standard_deviation), dim=1)
        return self.fc(pooled_features)


class MainModel(nn.Module):
    """Объединить генератор эмбеддингов и функцию потерь."""

    def __init__(self, model, trainfunc, **kwargs):
        super().__init__()
        self.speaker_model = model
        self.loss_function = trainfunc

    def forward(self, data, label=None):
        if data.ndim == 2:
            data = data.unsqueeze(1)
        batch_size = data.size(0)
        data = data.reshape(-1, data.size(-1))
        device = next(self.parameters()).device
        embeddings = self.speaker_model(data.to(device))
        embeddings = embeddings.reshape(batch_size, -1, embeddings.size(-1))
        embeddings = embeddings.mean(dim=1)

        if label is None:
            return embeddings

        label = label.reshape(-1).to(device=device, dtype=torch.long)
        return self.loss_function(embeddings, label)


def _validate_scheduler(scheduler):
    if scheduler is None:
        return
    if scheduler[1] not in {
        SCHEDULER_STEP_EPOCH,
        SCHEDULER_STEP_ITERATION,
    }:
        raise ValueError(ERROR_SCHEDULER_MODE.format(mode=scheduler[1]))


def train_network(
    train_loader,
    main_model,
    optimizer,
    scheduler,
    num_epoch,
    verbose=False,
):
    """Выполнить одну эпоху обучения и вернуть средние потери и точность."""
    _validate_scheduler(scheduler)
    main_model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    batch_count = 0
    example_count = 0

    for data, data_label in train_loader:
        optimizer.zero_grad(set_to_none=True)
        batch_loss, batch_accuracy = main_model(data, data_label)
        batch_loss.backward()
        optimizer.step()

        current_examples = int(data_label.numel())
        batch_count += 1
        example_count += current_examples
        total_loss += float(batch_loss.detach().cpu()) * current_examples
        total_accuracy += float(batch_accuracy.detach().cpu()) * current_examples

        if scheduler is not None and scheduler[1] == SCHEDULER_STEP_ITERATION:
            scheduler[0].step()

        if verbose:
            print(
                TRAIN_BATCH_MESSAGE.format(
                    epoch=num_epoch,
                    batch=batch_count,
                    learning_rate=optimizer.param_groups[0]["lr"],
                    loss=total_loss / example_count,
                    accuracy=total_accuracy / example_count,
                )
            )

    if batch_count == 0:
        raise ValueError(ERROR_EMPTY_LOADER)
    if scheduler is not None and scheduler[1] == SCHEDULER_STEP_EPOCH:
        scheduler[0].step()
    return total_loss / example_count, total_accuracy / example_count


def test_network(test_loader, main_model):
    """Оценить модель без вычисления градиентов."""
    main_model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    batch_count = 0
    example_count = 0

    with torch.no_grad():
        for data, data_label in test_loader:
            batch_loss, batch_accuracy = main_model(data, data_label)
            current_examples = int(data_label.numel())
            batch_count += 1
            example_count += current_examples
            total_loss += float(batch_loss.detach().cpu()) * current_examples
            total_accuracy += float(batch_accuracy.detach().cpu()) * current_examples

    if batch_count == 0:
        raise ValueError(ERROR_EMPTY_LOADER)
    return total_loss / example_count, total_accuracy / example_count
