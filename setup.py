from setuptools import setup, find_packages

setup(
    name="sneaker_bot_detection",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "tensorflow>=2.8.0",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "pydantic>=1.8.0",
        "tqdm>=4.62.0",
        "joblib>=1.0.0",
        "requests>=2.26.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.5",
            "jupyter>=1.0.0",
            "black>=21.5b0",
            "isort>=5.9.0",
            "flake8>=3.9.0",
        ],
    },
    description="Multi-Expert AI System for Sneaker Bot Detection",
    author="Arnav Khinvasara",
    author_email="akhinvasara@ucsd.edu",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.8",
)