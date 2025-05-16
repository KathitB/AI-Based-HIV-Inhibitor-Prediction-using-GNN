# utils.py

import os
import random
import numpy as np
import tensorflow as tf

def set_seed(seed=42):
    """
    Set seed for reproducibility across numpy, random, and TensorFlow.
    """
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def create_output_dir(path="outputs"):
    """
    Create a directory to store outputs like models, logs, etc.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def save_model(model, path="app/outputs/model"):
    """
    Save the DeepChem model to disk.
    """
    model.save_checkpoint(model_dir=path)

def load_model(model_class, path="app/outputs/model", **kwargs):
    """
    Load a DeepChem model from checkpoint.
    
    Args:
        model_class: e.g., dc.models.GraphConvModel
        path: Path to saved model directory
        **kwargs: Other arguments needed to re-instantiate the model

    Returns:
        Loaded model
    """
    model = model_class(**kwargs)
    model.restore(model_dir=path)
    return model
