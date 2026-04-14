"""
Dummy guidance and prompt processor for render-only tests.
"""

import threestudio
from threestudio.utils.base import BaseModule
from threestudio.utils.typing import *


@threestudio.register("dummy-guidance")
class DummyGuidance(BaseModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        # No-op guidance for render-only tests

    def __call__(self, *args, **kwargs):
        # Return dummy gradients
        return None


@threestudio.register("dummy-prompt-processor")
class DummyPromptProcessor(BaseModule):
    def __init__(self, cfg):
        super().__init__(cfg)

    def __call__(self, *args, **kwargs):
        # Return empty prompt utilities
        return {}
