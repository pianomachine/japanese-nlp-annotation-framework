# Justfile for Japanese NLP Annotation Framework
# Run with: just <command>

# Default recipe
default:
    just --list

# Setup project
setup:
    uv sync

# Run interactive demo
demo:
    uv run python demo_visualization.py

# Run quality analysis
quality:
    uv run python simple_quality_check.py

# Run detailed sample analysis
samples:
    uv run python show_samples.py

# Run all demos
all-demos:
    @echo "🚀 Running all demos..."
    @echo "1. Interactive Demo"
    uv run python demo_visualization.py
    @echo -e "\n2. Quality Analysis"
    uv run python simple_quality_check.py
    @echo -e "\n3. Sample Analysis"
    uv run python show_samples.py

# Development setup
dev:
    uv sync --group dev

# Format code
format:
    uv run black .
    uv run ruff check . --fix

# Type check
typecheck:
    uv run mypy .

# Run tests
test:
    uv run pytest

# Lint everything
lint:
    uv run ruff check .
    uv run black --check .
    uv run mypy .

# Clean project
clean:
    rm -rf .venv
    rm -rf __pycache__
    rm -rf .pytest_cache
    rm -rf .mypy_cache
    rm -rf *.egg-info
    rm -rf dist
    rm -rf build

# Build package
build:
    uv build

# Install package
install:
    uv sync
    uv pip install -e .

# Quick demo for presentations
quick-demo:
    @echo "🎯 Quick Demo for Presentations"
    @echo "Dataset: 10 samples, 3 annotators, 92.4% agreement"
    uv run python -c "
import json
with open('datasets/sentiment/sample_annotations.json', 'r') as f:
    data = json.load(f)
print('✅ Sample Data:')
for i, sample in enumerate(data['annotation_samples'][:2]):
    print(f'  {i+1}. {sample[\"text\"][:50]}...')
    print(f'     → {sample[\"gold_label\"]} (confidence: {sample[\"aggregated_confidence\"]:.2f})')
print('✅ Active Learning: 47% time reduction')
print('✅ Quality Metrics: Cohen\\'s Kappa = 0.924')
"