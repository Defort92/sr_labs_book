# Загрузка весов предобученной модели.

from collections import OrderedDict
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve

import torch

from constants import (
    ERROR_MODEL_DOWNLOAD,
    ERROR_MODEL_WEIGHTS,
    MODEL_DOWNLOADED_MESSAGE,
    MODEL_LOADED_MESSAGE,
    PRETRAINED_MODEL_FILENAME,
)


def load_model(model, lines, save_path, reload=False):
    """Загрузить файл весов и применить его к модели."""
    
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    for line in lines:
        url     = line.strip()
        outfile = url.split('/')[-1]
        output_path = save_path / outfile

        # Загружаем файл только при его отсутствии или явном обновлении.
        if not output_path.exists() or reload:
            try:
                urlretrieve(url, output_path)
            except (OSError, URLError) as error:
                raise RuntimeError(ERROR_MODEL_DOWNLOAD.format(url=url)) from error

        print(MODEL_DOWNLOADED_MESSAGE.format(filename=outfile))
        
    device = next(model.parameters()).device
    checkpoint = torch.load(
        save_path / PRETRAINED_MODEL_FILENAME,
        map_location=device,
        weights_only=False,
    )
    
    model_weight = OrderedDict()

    for key in checkpoint.keys():
        
        if '__S__' in key:
            model_weight[key[6:]] = checkpoint[key]
            
    if not model_weight:
        raise ValueError(ERROR_MODEL_WEIGHTS)
    model.load_state_dict(model_weight)
    print(MODEL_LOADED_MESSAGE)
    
    return model
