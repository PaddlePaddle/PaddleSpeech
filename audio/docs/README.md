# Build docs for PaddleAudio

Execute the following steps in **current directory**.

## 1. Install

`pip install Sphinx sphinx_rtd_theme`


## 2. Generate API docs

Generate API docs from doc string.

`sphinx-apidoc -fMeT -o source ../paddleaudio ../paddleaudio/utils --templatedir source/_templates`


## 3. Build

`sphinx-build source _html`


## 4. Preview

Open `_html/index.html` for page preview.
