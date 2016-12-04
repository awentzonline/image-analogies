#!/usr/bin/env python
from distutils.core import setup
from setuptools import find_packages

setup(
    name='neural-image-analogies',
    version='0.1.2',
    description='Generate image analogies with a deep neural network.',
    author='Adam Wentz',
    author_email='adam@adamwentz.com',
    url='https://github.com/awentzonline/image-analogies/',
    packages=find_packages(),
    scripts=[
        'scripts/make_image_analogy.py'
    ],
    install_requires=[
        'h5py>=2.5.0',
        'Keras>=1.0.0',
        'numpy>=1.10.4',
        'Pillow>=3.1.1',
        'PyYAML>=3.11',
        'scipy>=0.17.0',
        'scikit-learn>=0.17.0',
        'six>=1.10.0',
        'Theano>=0.8.2',
    ]
)
