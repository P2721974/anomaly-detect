# setup.py

from setuptools import setup, find_packages
from __version__ import __version__

setup(
    name="anomaly-detect",
    version=__version__,
    description="Anomaly-Based Threat Detection Pipeline using Machine Learning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="William Jecks",
    author_email="p2721974@my365.dmu.ac.uk",
    url="https://github.com/twisted-care/anomaly-detect",
    packages=find_packages(exclude=["tests*"]),
    include_package_data=True,
    install_requires=[
        "numpy==1.26.4",
        "pandas==2.2.3",
        "scikit-learn==1.3.2",
        "tensorflow==2.15.0",
        "joblib==1.4.0",
        "pyyaml==6.0.1",
        "scapy==2.5.0",
        "matplotlib==3.8.3",
        "seaborn==0.13.2",
        "pyshark==0.6",
        "requests==2.32.3",
        "tqdm==4.67.1",
    ],
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: >=3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "anomaly-detect=cli.main:main"
        ]
    },
)
