# 🚀 UV Usage Guide for JMMAF

This project is optimized for modern Python development with `uv` - the fast Python package manager.

## 📦 Installation

### 1. Install uv (if not already installed)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and setup the project
```bash
git clone https://github.com/pianomachine/japanese-nlp-annotation-framework
cd japanese-nlp-annotation-framework
uv sync
```

## 🎯 Running the Demos

### Basic Interactive Demo
```bash
uv run python demo_visualization.py
```
**Output**: Label distribution charts, active learning simulation, model benchmarks

### Quality Analysis
```bash
uv run python simple_quality_check.py
```
**Output**: Cohen's Kappa, agreement statistics, challenging samples

### Detailed Sample Analysis
```bash
uv run python show_samples.py
```
**Output**: Full annotation breakdown, Japanese-specific pattern analysis

## 🔧 Development Commands

### Install development dependencies
```bash
uv sync --group dev
```

### Run linting and formatting
```bash
uv run ruff check .
uv run black .
```

### Run type checking
```bash
uv run mypy .
```

### Run tests
```bash
uv run pytest
```

## 📊 Optional Dependencies

### Japanese NLP tools
```bash
uv sync --group japanese
```

### Web interface
```bash
uv sync --group web
```

### All dependencies
```bash
uv sync --group all
```

## 🎮 Command Line Tools

After installation, you can use:
```bash
# Quality evaluation
uv run jmmaf-evaluate

# Model benchmarking
uv run jmmaf-benchmark

# Active learning demo
uv run jmmaf-active-learn

# Interactive demo
uv run jmmaf-demo
```

## 🚀 Why uv?

### Speed
- **10-100x faster** than pip
- **Dependency resolution in seconds**
- **Parallel downloads**

### Reliability
- **Lock file** for reproducible builds
- **Conflict resolution**
- **Virtual environment management**

### Modern Python
- **pyproject.toml** native support
- **PEP 517/518** compliant
- **Python version management**

## 📋 Example Workflow

```bash
# Start development
uv sync --group dev

# Run all demos
uv run python demo_visualization.py
uv run python simple_quality_check.py
uv run python show_samples.py

# Format code
uv run black .
uv run ruff check . --fix

# Test
uv run pytest

# Build package
uv build
```

## 🔍 Troubleshooting

### Virtual Environment Issues
```bash
# Reset environment
rm -rf .venv
uv sync
```

### Dependency Conflicts
```bash
# Update lock file
uv lock --upgrade
```

### Python Version Issues
```bash
# Use specific Python version
uv sync --python 3.10
```

## 🌟 Benefits for xAI Application

Using `uv` demonstrates:
- ✅ **Modern Python practices**
- ✅ **Performance optimization**
- ✅ **Reproducible environments**
- ✅ **Professional development workflow**

This shows you're up-to-date with current Python ecosystem best practices!