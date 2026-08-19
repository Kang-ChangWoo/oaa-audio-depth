"""Shared library used by every entry point (train_*.py, eval.py, analysis/, viz/, tools/).

    core.data     get_data_module()        DATA_MODULE env -> data_0422 (Replica) | data_mp3d (Matterport3D)
    core.metrics  cos_lat, KEYS, BANDS, MetricAccumulator   cos-latitude-weighted per-image depth metrics
    core.ckpt     build, resolve_run, load_run              rebuild a model from a checkpoint's saved args
    core.evaluate evaluate                                  test-set evaluation of one run directory

Scripts import only `core.*` and `model.*`; they never import each other.
"""
