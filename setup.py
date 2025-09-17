#!/usr/bin/env python3
"""
Setup script for LLM Observability Framework
"""

from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open('requirements.txt', 'r') as f:
    requirements = [
        line.strip() 
        for line in f 
        if line.strip() and not line.startswith('#') and not line.strip().startswith('sqlite3')
    ]

setup(
    name="llm-observability-framework",
    version="0.1.0",
    description="A comprehensive framework for monitoring and observing Large Language Model (LLM) performance",
    author="Your Name",
    author_email="your.email@example.com",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=requirements,
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    entry_points={
        'console_scripts': [
            'llm-observability-dashboard=dashboard.streamlit_dashboard:main',
        ],
    },
)