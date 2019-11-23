import os
import json

from tensorflow.keras import models

from lib.model import DependencyParser


def load_pretrained_model(serialization_dir: str) -> models.Model:
    """
    Given serialization directory, returns: model loaded with the pretrained weights.
    """

    # Load Config
    config_path = os.path.join(serialization_dir, "config.json")
    model_path = os.path.join(serialization_dir, "model.ckpt.index")

    model_files_present = all([os.path.exists(path)
                               for path in [config_path, model_path]])
    if not model_files_present:
        raise Exception(f"Model files in serialization_dir ({serialization_dir}) "
                        f" are missing. Cannot load_the_model.")

    model_path = model_path.replace(".index", "")

    with open(config_path, "r") as file:
        config = json.load(file)

    # Load Model
    model = DependencyParser(**config)
    model.load_weights(model_path)

    return model
