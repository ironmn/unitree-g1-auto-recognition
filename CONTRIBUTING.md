# 持续迭代

1. 从 main 创建功能分支，按单个可验证能力提交 PR。避免数据采集、识别、策略和执行一次性混成一个变更。
2. 使用 Python 3.12 的独立开发环境，不将开发锁安装进已有 OrcaLab 桌面环境。`requirements-dev.txt` 是离线开发依赖锁；安装后用 `pip install --no-deps -e .`。
3. 修改依赖时同步 `pyproject.toml` 和相应锁文件。Windows 完整环境由 `requirements.in` → `requirement.txt`，核心开发环境由 pyproject → requirements-dev.txt。

```text
uv pip compile requirements.in --python-version 3.12 --python-platform x86_64-pc-windows-msvc --only-binary :all: -o requirement.txt
uv pip compile pyproject.toml --extra dev --universal --python-version 3.12 -o requirements-dev.txt
```

4. 提交前运行 Ruff、离线测试和包构建。改相机、SDK 或控制循环时必须附本机联调记录，或明确该部分尚未验证。
5. 数据语义变更先更新 docs/data_contract.md。必须保留状态顺序、单位、时钟来源、动作含义和观测/动作时间关系；破坏兼容时版本升级并提供迁移。
6. 数据、权重、日志和设备私有路径不提交 Git。训练集划分按运行/episode/场景而不是相邻帧；模型工件记录数据版本与代码提交。
7. OpenPI adapter、训练、远程推理、执行控制和结果判断各自独立开发。动作执行前确定动作空间，不将实测状态或仿真真值伪装成专家动作。

目前为 0.x 版本。可选元数据追加按补丁版本管理；核心字段/输入输出语义变更至少升级次版本，数据格式破坏另升级 schema_version。发布时更新 CHANGELOG，保留通过测试的版本和关联联调证据。
