"""Автоматические проверки третьей лабораторной работы."""

import tempfile
import unittest
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from audio_data import loadWAV
from constants import (
    ENCODER_ATTENTIVE_POOLING,
    ENCODER_STATISTICAL_POOLING,
    SAMPLE_RATE,
)
from exercises_blank import (
    MainModel,
    MaxoutLinear,
    ResNet,
    test_dataset_loader,
    test_network,
    train_dataset_loader,
    train_network,
)
from LossFunction import SoftmaxLoss
from ResNetBlocks import BasicBlock


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


class TinyEmbeddingModel(nn.Module):
    """Минимальная модель для быстрой проверки цикла обучения."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 4)

    def forward(self, data):
        return self.linear(data)


class SpeakerModelTests(unittest.TestCase):
    def test_load_wav_and_dataset_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            time = np.arange(1_000) / SAMPLE_RATE
            for name, frequency in (("a.wav", 180.0), ("b.wav", 260.0)):
                signal = np.sin(2 * np.pi * frequency * time).astype(np.float32)
                sf.write(directory_path / name, signal, SAMPLE_RATE)

            fragments = loadWAV(directory_path / "a.wav", 2, evalmode=True, num_eval=2)
            self.assertEqual(fragments.shape, (2, 560))

            data_list = ["диктор_1 a.wav", "диктор_2 b.wav"]
            train_dataset = train_dataset_loader(
                data_list,
                max_frames=2,
                train_path=directory,
            )
            test_dataset = test_dataset_loader(
                data_list,
                max_frames=2,
                test_path=directory,
            )
            self.assertEqual(len(train_dataset), 2)
            self.assertEqual(len(test_dataset), 2)
            self.assertEqual(train_dataset[0][0].shape, (1, 560))
            self.assertEqual({train_dataset[0][1], train_dataset[1][1]}, {0, 1})

    def test_maxout_selects_larger_activation(self):
        layer = MaxoutLinear(2, 1, bias=False)
        with torch.no_grad():
            layer.linear1.weight.copy_(torch.tensor([[1.0, 0.0]]))
            layer.linear2.weight.copy_(torch.tensor([[0.0, 1.0]]))
        output = layer(torch.tensor([[2.0, 3.0], [4.0, 1.0]]))
        torch.testing.assert_close(output, torch.tensor([[3.0], [4.0]]))

    def test_resnet_returns_fixed_embedding_for_both_pooling_types(self):
        audio = torch.randn(2, 4_000)
        for encoder_type in (
            ENCODER_STATISTICAL_POOLING,
            ENCODER_ATTENTIVE_POOLING,
        ):
            model = ResNet(
                BasicBlock,
                layers=(1, 1, 1, 1),
                activation=nn.ReLU,
                num_filters=(2, 4, 8, 16),
                nOut=12,
                encoder_type=encoder_type,
                n_mels=16,
            )
            model.eval()
            with torch.no_grad():
                embeddings = model(audio)
            self.assertEqual(embeddings.shape, (2, 12))
            self.assertTrue(torch.isfinite(embeddings).all())

    def test_training_and_testing_work_on_cpu(self):
        torch.manual_seed(7)
        data = torch.randn(12, 1, 8)
        labels = torch.tensor([0, 1] * 6)
        loader = DataLoader(TensorDataset(data, labels), batch_size=4)
        model = MainModel(
            TinyEmbeddingModel(),
            SoftmaxLoss(nOut=4, nClasses=2),
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        train_loss, train_accuracy = train_network(
            loader,
            model,
            optimizer,
            scheduler=None,
            num_epoch=1,
        )
        test_loss, test_accuracy = test_network(loader, model)
        self.assertGreater(train_loss, 0.0)
        self.assertGreater(test_loss, 0.0)
        self.assertGreaterEqual(train_accuracy, 0.0)
        self.assertLessEqual(test_accuracy, 100.0)


if __name__ == "__main__":
    unittest.main()
