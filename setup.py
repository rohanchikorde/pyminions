from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pyminions",
    version="0.1.0",
    author="Rohan Chikorde",
    author_email="rohan.chikorde@derivco.com",
    description="A comprehensive model evaluation framework for machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rohanchikorde/pyminions",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[
        'scikit-learn>=1.0.2',
        'yellowbrick>=1.5',
        'mlflow>=2.3.0',
        'pandas>=1.5.0',
        'numpy>=1.21.0',
        'matplotlib>=3.5.0',
        'seaborn>=0.12.0',
        'jinja2>=3.0.0'
    ],
    package_data={
        'pyminions': ['templates/*.html'],
    },
    include_package_data=True,
)
