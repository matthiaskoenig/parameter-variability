# Release information

# Install dev dependencies:
```bash
# install core dependencies
uv sync
# install dev dependencies
uv pip install -r pyproject.toml --extra dev
uv tool install tox --with tox-uv
```

## Testing with tox
Run single tox target
```bash
tox r -e py314
```
Run all tests in parallel
```bash
tox run-parallel
```

# Setup pre-commit
```bash
uv pip install pre-commit
pre-commit install
pre-commit run
```
