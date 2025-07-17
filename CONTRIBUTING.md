# Contributing to Japanese Multi-Modal Annotation Framework (JMMAF)

Thank you for your interest in contributing to JMMAF! This project aims to provide high-quality Japanese language annotation tools and datasets for the research community.

## 🚀 Quick Start

1. **Fork the repository** and clone it locally
2. **Create a virtual environment**: `python -m venv jmmaf-env`
3. **Activate the environment**: `source jmmaf-env/bin/activate` (Linux/Mac) or `jmmaf-env\Scripts\activate` (Windows)
4. **Install dependencies**: `pip install -r requirements.txt`
5. **Run tests**: `python -m pytest tests/`

## 🎯 Ways to Contribute

### 1. Data Annotation
- **Expand existing datasets**: Add more Japanese text samples with annotations
- **Create new domain datasets**: Restaurant reviews, social media, academic papers
- **Improve annotation quality**: Review existing annotations for consistency

### 2. Code Contributions
- **Bug fixes**: Fix issues in annotation tools or evaluation scripts
- **Feature requests**: Add new active learning strategies or quality metrics
- **Performance improvements**: Optimize model inference or annotation workflows

### 3. Documentation
- **Translation**: Translate guidelines to other languages
- **Examples**: Add more usage examples and tutorials
- **API documentation**: Improve code documentation and docstrings

### 4. Research
- **New annotation schemes**: Design annotation frameworks for new NLP tasks
- **Evaluation metrics**: Develop Japanese-specific evaluation metrics
- **Baseline models**: Add new model architectures for benchmarking

## 📋 Annotation Guidelines

### For Data Annotators
1. **Read the guidelines carefully**: `/annotation_guidelines/sentiment_analysis_guideline.md`
2. **Follow the quality standards**: Aim for >90% inter-annotator agreement
3. **Document edge cases**: Report ambiguous samples for guideline updates
4. **Use the annotation tools**: Leverage the web interface for efficient annotation

### Quality Requirements
- **Consistency**: Follow established annotation patterns
- **Documentation**: Explain reasoning for difficult cases
- **Coverage**: Ensure balanced representation across domains and sentiment classes

## 🧪 Code Standards

### Python Style
- **Code formatting**: Use `black` for consistent formatting
- **Linting**: Run `flake8` to check for style issues
- **Type hints**: Include type annotations for function parameters and returns
- **Documentation**: Write clear docstrings for all functions and classes

### Testing
- **Unit tests**: Write tests for new functions using `pytest`
- **Integration tests**: Test end-to-end workflows
- **Performance tests**: Ensure new features don't slow down existing functionality

### Example Code Structure
```python
def calculate_agreement(annotations: Dict[str, List[str]]) -> float:
    """
    Calculate inter-annotator agreement using Cohen's Kappa.
    
    Args:
        annotations: {annotator_id: [labels]}
        
    Returns:
        Cohen's Kappa coefficient
        
    Example:
        >>> annotations = {"A001": ["POS", "NEG"], "A002": ["POS", "NEU"]}
        >>> agreement = calculate_agreement(annotations)
        >>> print(f"Agreement: {agreement:.3f}")
    """
    # Implementation here
    pass
```

## 📊 Benchmarking

### Adding New Models
1. **Model integration**: Add model to `models/benchmark_results.py`
2. **Performance testing**: Run benchmark on test dataset
3. **Documentation**: Update README with new results

### Evaluation Metrics
- **Accuracy**: Overall classification accuracy
- **F1 Score**: Weighted F1 for imbalanced datasets
- **Speed**: Inference time and throughput
- **Resource usage**: Memory and GPU requirements

## 🤝 Community

### Communication
- **Issues**: Use GitHub issues for bug reports and feature requests
- **Discussions**: Join our research discussions in GitHub Discussions
- **Email**: Contact lead maintainer at ryo.yanagisawa@ogata-lab.org

### Research Collaboration
- **Paper submissions**: Coordinate on research publications
- **Conference presentations**: Share presentation materials
- **Dataset sharing**: Contribute to community datasets

## 📄 License

By contributing to JMMAF, you agree that your contributions will be licensed under the MIT License.

## 🙏 Recognition

Contributors will be acknowledged in:
- **README.md**: Listed in contributors section
- **Research papers**: Co-authorship opportunities for significant contributions
- **Conference presentations**: Recognition in acknowledgments

## 📝 Submission Process

1. **Create a feature branch**: `git checkout -b feature/your-feature-name`
2. **Make your changes**: Follow coding standards and add tests
3. **Run quality checks**: `black`, `flake8`, `mypy`, `pytest`
4. **Commit your changes**: Use clear, descriptive commit messages
5. **Push to your fork**: `git push origin feature/your-feature-name`
6. **Create a pull request**: Describe your changes and their impact

### Pull Request Template
```markdown
## Summary
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Breaking changes documented
```

## 🔄 Review Process

1. **Automated checks**: CI/CD pipeline runs tests and quality checks
2. **Peer review**: At least one maintainer reviews the code
3. **Testing**: Manual testing for complex features
4. **Merge**: Approved PRs are merged into main branch

Thank you for helping improve Japanese NLP research! 🇯🇵✨