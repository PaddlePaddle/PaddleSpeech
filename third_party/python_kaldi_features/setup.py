try:
    from setuptools import setup #enables develop
except ImportError:
    from distutils.core import setup

with open("requirements.txt", encoding="utf-8-sig") as f:
    requirements = f.readlines()

setup(name='paddlespeech_feat',
      version='0.1.0',
      description='python speech feature extraction in paddlespeech',
      install_requires=requirements,
      author="PaddlePaddle Speech and Language Team",
      author_email="paddlesl@baidu.com",
      license='MIT',
      url='https://github.com/PaddlePaddle/PaddleSpeech',
      packages=['python_speech_features'],
    )
