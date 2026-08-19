"""Data-module selection. Both modules expose the same interface:
    loader(split, bs, shuffle, nw, mode) / wave_loader(...) / IN_CH[mode] / POSES[mode] / RotSet."""
import importlib
import os

DEFAULT = "data_mp3d"


def get_data_module(default=DEFAULT):
    """Return the data module named by $DATA_MODULE (data_0422 = Replica, data_mp3d = Matterport3D)."""
    return importlib.import_module(os.environ.get("DATA_MODULE", default))
