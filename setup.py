import setuptools
import os

with open("README.md", "r") as fh:
    long_description = fh.read()

INSTALL_REQUIRES = sorted(set(
    line.partition('#')[0].strip()
    for line in open(os.path.join(os.path.dirname(__file__), 'requirements.txt'))
) - {''})

setuptools.setup(
    name="keras-explain",
    version="0.0.1",
    author="Primoz Godec",
    author_email="primoz492@gmail.com",
    description="Explanation toolbox for Keras models. ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/primozgodec/keras-explain",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=INSTALL_REQUIRES
)
