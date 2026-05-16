from setuptools import setup, find_packages

setup(
    name="demand-forecasting-mlops",
    version="1.0.0",
    description="Production-ready MLOps pipeline for e-commerce demand forecasting",
    author="Narendra Tekale",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "pandas>=2.1.4",
        "numpy>=1.26.4",
        "scikit-learn>=1.4.0",
        "xgboost>=2.0.3",
        "prophet>=1.1.5",
        "mlflow>=2.10.2",
        "fastapi>=0.109.2",
        "uvicorn>=0.27.1",
        "pyyaml>=6.0.1",
    ],
)
