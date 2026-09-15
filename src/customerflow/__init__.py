"""CustomerFlow: a reproducible customer value modelling pipeline."""

from .data import FEATURES, TARGET, load_source, prepare_frame
from .model import train_model

__all__ = ["FEATURES", "TARGET", "load_source", "prepare_frame", "train_model"]
