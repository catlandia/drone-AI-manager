"""Setup script for drone-ai package."""

from setuptools import setup, find_packages

setup(
    name="drone-ai",
    version="0.1.0",
    description="Drone AI delivery environment with mission planning",
    author="Drone AI Team",
    python_requires=">=3.8",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.20.0",
        "gymnasium>=0.28.0",
    ],
    extras_require={
        "viz": ["pygame>=2.0.0"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "drone-demo=drone_ai.demo:main",
            "drone-learn=drone_ai.learning_sequence:main",
        ],
    },
)
