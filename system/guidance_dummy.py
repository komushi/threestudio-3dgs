"""
Dummy guidance for render-only tests.
"""

import threestudio
from dataclasses import dataclass, field
from typing import Any

from threestudio.utils.base import BaseModule
from threestudio.utils.typing import *


@threestudio.register("dummy-guidance")
class DummyGuidance(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: str = ""
        # Accept any config fields without error

    cfg: Config

    def __init__(self, cfg):
        super().__init__(cfg)
        # No-op guidance for render-only tests

    def __call__(self, *args, **kwargs):
        # Return dummy gradients
        return None
