from setuptools import setup, find_packages

setup(
    name="label-metric",
    version="0.1",
    description="",
    author="Haokun Tian",
    author_email="haokun.tian@qmul.ac.uk",
    include_package_data=True,
    packages=find_packages(),  # utils subpkg
    url="https://github.com/tiianhk/label-metric",
    install_requires=[
        "torch",
        "torchaudio<=2.8",
        "torchmetrics",
        "pytorch-lightning",
        "pytorch-metric-learning",
        "anytree",
    ],
)