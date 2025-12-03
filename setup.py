"""
Setup script for learning-optimization package.

This is maintained for backwards compatibility with older pip versions.
Modern installations should use pyproject.toml.
"""
from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        name="learning-optimization",
        packages=find_packages(include=["optimizers", "optimizers.*", "data", "data.*"]),
        package_data={
            "optimizers": ["*.md", "*.txt"],
            "data": ["*.mat", "*.csv"],
        },
    )
