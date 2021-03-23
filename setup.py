import setuptools

setuptools.setup(
    name="batchlogit",
    version="0.0.1",
    url="https://github.com/frankier/batchlogit",
    author="Frankie Robertson",
    description="Experiment/benchmark for training in parallel many logistic regression models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=[
        "more_itertools>=8.6.0",
        "scikit-learn>=0.24.1",
        "click>=7.1.2",
        "click-log>=0.3.2",
        "pytorch>=1.8.0",
    ],
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
    ],
)
