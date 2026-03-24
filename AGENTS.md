# Agent Guidelines for MindONE

This file provides coding guidelines and instructions for AI agents working in this repository.

## Project Overview

MindONE is a MindSpore-based implementation of Hugging Face Diffusers and Transformers, providing 350+ state-of-the-art ML models for text, vision, audio, video, and multimodal generation. The codebase is adapted from Hugging Face libraries with modifications for MindSpore compatibility.

## Build/Lint/Test Commands

### Installation
```bash
# Install in development mode with all dependencies
pip install -e ".[dev]"

# Install with specific optional dependencies
pip install -e ".[lint]"      # pre-commit hooks
pip install -e ".[tests]"     # pytest, parameterized, pytest-cov
pip install -e ".[docs]"      # mkdocs and plugins
pip install -e ".[training]"  # training dependencies
```

### Pre-commit Hooks (Required Before Commit)
```bash
# Run all linters
pre-commit run --show-diff-on-failure --color=always --all-files

# Install hooks to run automatically on git commit
pre-commit install

# Run specific hook
pre-commit run black
pre-commit run isort
pre-commit run flake8
```

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=mindone --cov-report=html

# Run specific test file
pytest tests/diffusers_tests/test_schedulers.py

# Run specific test class
pytest tests/diffusers_tests/pipelines/wan/test_wan.py::WanPipelineFastTests

# Run specific test function
pytest tests/diffusers_tests/pipelines/wan/test_wan.py::WanPipelineFastTests::test_inference

# Run tests matching a pattern
pytest -k "test_inference"

# Run with verbose output
pytest -v

# Run with parallel execution (if pytest-xdist installed)
pytest -n auto
```

### Building Documentation
```bash
mkdocs build      # Build static site
mkdocs serve      # Serve locally for development
```

---

## Code Style Guidelines

### General Style
- **Line Length**: 120 characters (enforced by Black)
- **Python Version**: 3.9+ (see `pyproject.toml`)
- **Encoding**: UTF-8 with `# coding=utf-8` header for files with non-ASCII

### Imports (isort Configuration)
Use the following import order:
```python
# 1. Future imports
from __future__ import annotations

# 2. Standard library
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# 3. Third-party (alphabetically within section)
import numpy as np
import PIL.Image
import torch

# 4. MindSpore
import mindspore as ms
from mindspore import mint, nn, ops

# 5. First-party (mindone modules)
from mindone.diffusers import ...
from mindone.utils import ...

# 6. Local imports (from . or ..)
from .utils import ...
```

Import sections defined in `pyproject.toml`:
```toml
[tool.isort]
sections = ["FUTURE", "STDLIB", "THIRDPARTY", "MINDSPORE", "FIRSTPARTY", "LOCALFOLDER"]
profile = "black"
line_length = 120
```

### Naming Conventions
- **Classes**: `CamelCase` (e.g., `DiffusionPipeline`, `ModelMixin`)
- **Functions/Methods**: `snake_case` (e.g., `get_scheduler`, `load_state_dict`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_SEQUENCE_LENGTH`, `THRESHOLD_FP32`)
- **Variables**: `snake_case` (e.g., `batch_size`, `output_hidden_states`)
- **Private methods/attributes**: Leading underscore (e.g., `_internal_method`, `_private_attr`)
- **Type variables**: Capitalized (e.g., `T`, `OutputT`)

### Type Annotations
- Use type hints for all public functions and methods
- Use `Union` instead of `|` for Python 3.9 compatibility
- Use `Optional[X]` instead of `X | None` for compatibility
- Common imports:
```python
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from typing_extensions import Self  # For methods returning class type
```

### Docstrings
Follow Google-style docstrings:
```python
def get_scheduler(
    scheduler_name: str,
    num_training_steps: int,
    num_warmup_steps: int = 0,
) -> Any:
    """Retrieve a scheduler by name.

    Args:
        scheduler_name: Name of the scheduler to retrieve.
        num_training_steps: Total number of training steps.
        num_warmup_steps: Number of warmup steps. Defaults to 0.

    Returns:
        The scheduler instance.

    Raises:
        ValueError: If scheduler_name is not recognized.
    """
```

### Error Handling
- Use specific exceptions (not bare `except:`)
- Log errors with module logger:
```python
logger = logging.get_logger(__name__)

try:
    result = risky_operation()
except SpecificError as e:
    logger.warning(f"Operation failed with specific error: {e}")
    raise AnotherError("Context message") from e
```

### MindSpore-Specific Patterns

#### MindSpore Imports
```python
import mindspore as ms
from mindspore import mint, nn, ops
from mindspore.common.initializer import initializer, XavierUniform
```

#### Cell Class (Neural Network Layers)
```python
class MyModel(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.Dense(in_channels, out_channels)
        self.dropout = nn.Dropout(keep_prob=0.1)

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        x = self.layer(x)
        return self.dropout(x)
```

#### Parameter Initialization
```python
self.weight = ms.Parameter(
    ms.Tensor(shape, dtype=dtype),
    name="weight"
)
# Or use initializer
self.weight = ms.Parameter(
    initializer(XavierUniform(), shape, dtype=dtype),
    name="weight"
)
```

#### Context Manager for Execution Mode
```python
from mindone.diffusers.utils.mindspore_utils import pynative_context

with pynative_context():
    output = model(input_tensor)
```

### Flake8 Configuration
Max line length: 160 (from `.flake8`), but prefer 120 for new code.

Ignored rules: `E203` (whitespace before ':'), `E731` (lambda assignment)

---

## Testing Guidelines

### Test Structure
- Tests use `unittest.TestCase` with `ddt` (data-driven tests) for parameterized testing
- Use `pytest` as the test runner
- Test files go in `tests/` directory, mirroring source structure

### Example Test Pattern
```python
import unittest
import pytest
from ddt import data, ddt, unpack
import mindspore as ms

test_cases = [
    {"mode": ms.PYNATIVE_MODE, "dtype": "float32"},
    {"mode": ms.PYNATIVE_MODE, "dtype": "float16"},
]

@ddt
class MyModelTests(unittest.TestCase):
    def get_dummy_components(self):
        components = {...}
        return components

    def get_dummy_inputs(self):
        inputs = {...}
        return inputs

    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        ms.set_context(mode=mode)
        # test implementation
```

### Test Thresholds
```python
THRESHOLD_FP16 = 5e-2
THRESHOLD_FP32 = 5e-3
THRESHOLD_PIXEL = 20.0
AUDIO_THRESHOLD_FP16 = 1e-1
AUDIO_THRESHOLD_FP32 = 5e-1
```

### Skipping Tests
```python
from mindone.diffusers.utils.testing_utils import slow

@unittest.skipIf(condition, "reason")
class SlowTests:
    pass

@slow
def test_large_model():
    """Tests that require model downloads."""
    pass
```

### Random Seed for Reproducibility
```python
def set_seed(seed: int = 42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    ms.set_seed(seed)
```

---

## Git Workflow

### Commit Messages
- Use clear, descriptive commit messages
- Reference issues when applicable: `Fix #123: description`

### Branch Naming
- Feature: `feature/description`
- Bugfix: `fix/description`
- Hotfix: `hotfix/description`

### Pull Request Checklist
- [ ] Run pre-commit hooks on all changed files
- [ ] Add/update tests for new functionality
- [ ] Update documentation if needed
- [ ] Ensure all tests pass

---

## File Header Template
For new source files:
```python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
#
# This code is adapted from https://github.com/huggingface/diffusers
# with modifications to run diffusers on mindspore.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
```

---

## Key Dependencies

- **mindspore**: Primary deep learning framework
- **transformers**: 4.57.1 compatibility
- **diffusers**: 0.35.2 compatibility (via mindone)
- **torch**: Used in tests for PyTorch comparison
- **numpy**: <2.0 for compatibility

---

## Additional Resources

- [MindSpore Documentation](https://www.mindspore.cn/)
- [HuggingFace Diffusers](https://github.com/huggingface/diffusers)
- [Contributing Guide](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)
