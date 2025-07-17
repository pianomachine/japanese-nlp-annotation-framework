#!/usr/bin/env python3
"""
Setup script for Japanese Multi-Modal Annotation Framework (JMMAF)
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="jmmaf",
    version="1.0.0",
    author="Ryo Yanagisawa",
    author_email="ryo.yanagisawa@ogata-lab.org",
    description="A comprehensive framework for high-quality Japanese language data annotation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ryo-yanagisawa/japanese-nlp-annotation-framework",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Natural Language :: Japanese",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=21.7b0",
            "flake8>=3.9.0",
            "mypy>=0.910",
            "pre-commit>=2.15.0",
        ],
        "docs": [
            "sphinx>=4.2.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "jmmaf-evaluate=evaluate.quality_metrics:main",
            "jmmaf-benchmark=models.benchmark_results:main",
            "jmmaf-active-learn=annotation_tools.active_learning.uncertainty_sampling:main",
        ],
    },
    include_package_data=True,
    package_data={
        "jmmaf": [
            "annotation_guidelines/*.md",
            "annotation_guidelines/*.pdf",
            "datasets/*/sample_*.json",
        ],
    },
)