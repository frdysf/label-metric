from setuptools import setup

setup(
    name="label-metric",
    version="0.1",
    description="",
    author="Haokun Tian",
    author_email="haokun.tian@qmul.ac.uk",
    include_package_data=True,
    packages=['label_metric'],
    url="https://github.com/tiianhk/label-metric",
    install_requires=[
        "torch",
        "torchaudio",
        "torchmetrics",
        "pytorch-lightning",
        "pytorch-metric-learning",
        "anytree",
    ],
)