# Build docs for PaddleAudio

## 1. Install

`pip install Sphinx`
`pip install sphinx_rtd_theme`


## 2. Generate API docs

Exclude `paddleaudio.utils`

`sphinx-apidoc -fMeT -o source ../paddleaudio ../paddleaudio/utils --templatedir source/_templates`


## 3. Build

`sphinx-build source _html`