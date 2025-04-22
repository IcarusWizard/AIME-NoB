import os
from setuptools import setup
from setuptools import find_packages
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'aime_nob'))

# find the version 
version_file = os.path.join(os.path.dirname(__file__), 'aime_nob', 'configs', 'version', 'default.yaml')
with open(version_file, 'r') as f:
    version_text = f.read()
__version__ = version_text.split(':')[-1].strip()

assert sys.version_info.major == 3, \
    "This repo is designed to work with Python 3." \
    + "Please install it before proceeding."

setup(
    name='aime_nob',
    author="Xingyuan Zhang",
    author_email="xingyuan.zhang@tum.de",
    packages=find_packages(),
    version=__version__,
    install_requires=[
        'torch',
        'torchvision',
        'numpy',
        'einops',
        'dm_control',
        'mujoco',
        'gym',
        'matplotlib',
        'tensorboard',
        'tqdm',
        'moviepy',
        'imageio==2.27',
        'hydra-core',
        'timm',
    ],
    url="https://github.com/IcarusWizard/AIME-NoB",
)