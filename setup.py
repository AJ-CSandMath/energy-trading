"""
Setup file for Energy Trading package.
Allows installation in editable mode: pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="energy_trading",
    version="0.1.0",
    description="Energy Trading & Portfolio Optimization System",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "pandas",
        "numpy",
        "scipy",
        "matplotlib",
        "seaborn",
        "statsmodels",
        "pyarrow",
        "pyyaml",
        "python-dotenv",
        "scikit-learn",
        "xgboost",
        "tensorflow",
        "keras",
        "h5py",
        "cvxpy",
        "streamlit",
        "plotly",
        "pytest",
        "requests",
    ],
)
