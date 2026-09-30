"""E139 shadow of the repo model package: exposes OAAv2Depth from the patched oaa.py only.

PYTHONPATH puts /root/storage/e139_code ahead of the repo, so `import model` resolves here. The
trainer only needs OAAv2Depth (train_oaa_e144.py:56), so the repo __init__'s sibling imports
(batvision, pretrained, beyond_i2d, echoscan) are deliberately NOT re-exported: pulling them in
would require copying those modules too and would widen the diff for no reason. `core.*` still
resolves to the repo because e139_code has no core/.
"""
from .oaa import OAAv2Depth
