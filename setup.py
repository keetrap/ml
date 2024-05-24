from setuptools import setup, find_packages

setup(
    name="Mlops",
    version="0.1.0",
    author="Prateek Kamboj",
    author_email="parteekkamboj112@gmail.com",
    url="https://github.com/keetrap/ml",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Linux",
    ],
    python_requires='>=3.6',
    install_requires=[
        "numpy>=1.19.5",
        "pandas>=1.1.5",
        "scikit-learn>=0.24.1",
        "matplotlib>=3.3.4",
        "seaborn>=0.11.1",
        "flask",
        "mlflow",
        "dvc",
        "xgboost",
        "tox",
        "scikit-learn"
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "flake8>=3.8.4",
            "black>=20.8b1",
        ],
    },
    packages=find_packages(),
    
)
