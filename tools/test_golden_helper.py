#
# Ko, (Baczynski-Bruce) 2026; Apache 2.0
#
"""Regression tests for pinned reference generation preconditions."""

from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from golden_helper import generate_real_checkpoint_reference


class RealReferenceRuntimeTests(unittest.TestCase):
    def test_disabled_onednn_fails_before_checkpoint_loading(self) -> None:
        torch_module = types.ModuleType("torch")
        torch_module.backends = types.SimpleNamespace(
            mkldnn=types.SimpleNamespace(is_available=lambda: True, enabled=False)
        )

        model_loader = Mock(side_effect=AssertionError("checkpoint loading must not start"))
        transformers_module = types.ModuleType("transformers")
        transformers_module.AutoModelForCausalLM = types.SimpleNamespace(from_pretrained=model_loader)

        with tempfile.TemporaryDirectory() as checkpoint_dir:
            with patch.dict(
                "sys.modules",
                {"torch": torch_module, "transformers": transformers_module},
            ):
                with self.assertRaisesRegex(RuntimeError, "oneDNN is available but disabled"):
                    generate_real_checkpoint_reference(Path(checkpoint_dir))

        model_loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
