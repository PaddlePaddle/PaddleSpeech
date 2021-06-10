import setuptools
import codecs
import os.path

with open("README.md", "r") as fh:
    long_description = fh.read()

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()    
    
def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")    
    
setuptools.setup(
    name="nnAudio", # Replace with your own username
    version=get_version("nnAudio/__init__.py"),
    author="KinWaiCheuk",
    author_email="u3500684@connect.hku.hk",
    description="A fast GPU audio processing toolbox with 1D convolutional neural network",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KinWaiCheuk/nnAudio",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
