# Agent Guidelines for MindONE

MindONE is a MindSpore-based implementation of HuggingFace Diffusers/Transformers for 350+ ML models.

## Build/Lint/Test Commands

### Installation
```bash
pip install -e ".[dev]"      # All dependencies
pip install -e ".[lint]"     # Pre-commit hooks
pip install -e ".[tests]"    # pytest, parameterized
```

### Pre-commit (Required Before Commit)
```bash
pre-commit run --show-diff-on-failure --color=always --all-files
pre-commit install  # Auto-run on git commit
```

### Running Tests
```bash
pytest                                    # All tests
pytest --cov=mindone --cov-report=html   # With coverage
pytest tests/file.py                     # Single file
pytest tests/file.py::ClassName          # Single class
pytest tests/file.py::ClassName::func    # Single function
pytest -k "pattern"                      # By pattern
pytest -v -n auto                         # Verbose + parallel
```

---

## Code Style

### General
- **Line Length**: 120 chars (Black)
- **Python**: 3.9+
- **Encoding**: UTF-8 with `# coding=utf-8` header for non-ASCII

### Imports (isort order)
```python
from __future__ import annotations

# Stdlib
import os
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party (alphabetical)
import numpy as np
import torch

# MindSpore
import mindspore as ms
from mindspore import mint, nn, ops

# First-party
from mindone.diffusers import ...
from mindone.utils import ...

# Local
from .utils import ...
```

### Naming Conventions
- Classes: `CamelCase` (e.g., `DiffusionPipeline`)
- Functions/Methods: `snake_case` (e.g., `get_scheduler`)
- Constants: `UPPER_SNAKE_CASE`
- Private: Leading underscore (`_private_method`)

### Type Annotations
- Use `Union` instead of `|` (Python 3.9 compat)
- Use `Optional[X]` instead of `X | None`
- Import `Self` from `typing_extensions`

### Docstrings
Google-style. Example:
```python
def func(arg: int, opt: str = "default") -> bool:
    """Short description.

    Args:
        arg: Description.
        opt: Optional param. Defaults to "default".

    Returns:
        What is returned.

    Raises:
        ValueError: When invalid.
    """
```

### Error Handling
- Use specific exceptions (no bare `except:`)
- Log with `logger = logging.get_logger(__name__)`

---

## MindSpore Patterns

```python
import mindspore as ms
from mindspore import mint, nn, ops
from mindspore.common.initializer import initializer, XavierUniform

class MyModel(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.Dense(in_channels, out_channels)

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        return self.layer(x)

# Parameter initialization
self.weight = ms.Parameter(initializer(XavierUniform(), shape, dtype), name="weight")

# Pynative context for dynamic shapes
from mindone.diffusers.utils.mindspore_utils import pynative_context
with pynative_context():
    output = model(input_tensor)
```

---

## Testing

```python
import unittest
from ddt import data, ddt, unpack
import mindspore as ms

test_cases = [{"mode": ms.PYNATIVE_MODE, "dtype": "float32"}]

@ddt
class MyTests(unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        ms.set_context(mode=mode)
        # test code

# Test thresholds
THRESHOLD_FP16 = 5e-2
THRESHOLD_FP32 = 5e-3
```

### Random Seed
```python
def set_seed(seed: int = 42):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    ms.set_seed(seed)
```

---

## Git Workflow
- Commits: `Fix #123: description`
- Branches: `feature/desc`, `fix/desc`, `hotfix/desc`
- PR checklist: Run pre-commit, add tests, update docs

---

## File Header (new files)
```python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
```

---

## Key Dependencies
- **mindspore**: Primary framework
- **transformers**: 4.57.1
- **diffusers**: 0.35.2 (via mindone)
- **numpy**: <2.0
