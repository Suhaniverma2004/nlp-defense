# Development and test environment

## Supported Python version

Python 3.10 or newer is required. The shared Unicode/alignment modules use
Python 3.10 union type syntax. The repository's CI currently runs Python 3.13,
and the documented clean development environment has been verified with
CPython 3.14.2; those versions do not establish compatibility with versions
older than the declared 3.10 minimum.

## Verified environment

The repository was checked with 64-bit CPython 3.14.2 on Windows. A clean
environment installed from the existing `requirements.txt` resolved these
versions:

- PyTorch 2.10.0 (`torch.__version__` reports `2.10.0+cpu`)
- Transformers 5.2.0
- Tokenizers 0.22.2
- NumPy 2.4.2
- pandas 3.0.1
- scikit-learn 1.8.0
- pytest 9.0.2

`pip check` reported no broken requirements. The CPU PyTorch wheel imports
`torch.nn` successfully. No dependency pins were changed for this milestone.

## Create the environment on Windows

Use the Python 3.14 interpreter that will run the project. This procedure
creates a fresh environment under the ignored `.venv` directory and installs
the repository's pinned requirements:

```powershell
python -m venv --without-pip .venv\milestone1-clean
python -m pip --python .venv\milestone1-clean install -r requirements.txt
```

Run the test suite with that environment:

```powershell
.\.venv\milestone1-clean\Scripts\python.exe -m pytest -q
```

The `--without-pip` step is useful on the verified host because the original
venv's `ensurepip` bootstrap failed. The base interpreter's pip supports the
`--python` target option and installs into the new environment without relying
on the damaged environment or changing the pinned requirements.

## Diagnosing the original PyTorch import

The original `.venv` used CPython 3.14.2, but its `site-packages\torch` was an
incomplete directory containing only a few subdirectories. It had no
`torch\__init__.py` and no `torch-*.dist-info` metadata. Consequently,
`find_spec("torch")` returned a namespace-package spec with no loader, and
`import torch` exposed no `torch.nn` module. This was an incomplete local
installation, not a conflict between the pinned PyTorch version and the
project's Python executable.

The clean environment installs the pinned wheel and resolves the import. Keep
using its interpreter for both package management and tests so the `python`
and `pip` commands refer to the same environment.
