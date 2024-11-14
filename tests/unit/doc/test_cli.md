# test CLI 测试文档

 该文档为 CLI 测试说明，该测试目前覆盖大部分 paddlespeech 中的 CLI 推理。该 CI 建立后用于快速验证修复是否正确。

 # 测试流程
 ## 1. 环境安装

 CI 重建时在已有通过版本 paddlepaddle-gpu==2.5.1, paddlepseech==develop 下运行。

 CI 重建后在 paddlepaddle-gpu==develop, paddlepseech==develop 下运行。
 
 ### 其他相关依赖

 gcc >= 4.8.5,
 python >= 3.8

 ## 2. 功能测试

 在 repo 的 tests/unit/cli 中运行：

  ```shell

  source path.sh
  bash test_cli.sh

  ```
## 3. 预期结果

 输出 "Test success"，且运行过程中无报错或 Error 即为成功。
