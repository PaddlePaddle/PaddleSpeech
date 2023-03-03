# 💡 paddlespeech 提交代码须知

### Discussed in https://github.com/PaddlePaddle/PaddleSpeech/discussions/1326

<div type='discussions-op-text'>

<sup>Originally posted by **yt605155624** January 12, 2022</sup>
1. 写完代码之后可以用我们的 pre-commit 检查一下代码格式，注意只改自己修改的代码的格式即可，其他的代码有可能也被改了格式，不要 add 就好
```
pip install pre-commit
pre-commit run --file 你修改的代码
```
2. 提交 commit 中增加必要信息跳过不必要的 CI
- 提交 asr 相关代码
```text
git commit -m "xxxxxx, test=asr"
```
- 提交 tts 相关代码
```text
git commit -m "xxxxxx, test=tts"
```
- 仅修改文档
```text
git commit -m "xxxxxx, test=doc"
```
注意：
1. 虽然跳过了 CI，但是还要先排队排到才能跳过，所以非自己方向看到 pending 不要着急 🤣
2. 在 `git commit --amend` 的时候才加 `test=xxx` 可能不太有效
3. 一个 pr 多次提交 commit 注意每次都要加 `test=xxx`，因为每个 commit 都会触发 CI
4. 删除 python 环境中已经安装好的的 paddlespeech，否则可能会影响 import paddlespeech 的顺序</div>
