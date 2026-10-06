# 新环境迁移与快速开始

先克隆本仓库，后续命令从项目根目录运行。工具脚本使用当前 Python；不要依赖旧电脑的用户目录。依赖固定版本见 `configs/upstream_versions.json`。

## 选择环境

| 用途 | 环境 | 依赖来源 |
|---|---|---|
| 观测与离线开发测试 | Python 3.12；Windows 或 Linux | `requirements-dev.txt` + 本包 |
| 仿真动作采集 | Windows OrcaLab 26.8.2，官方独立采集环境 | 官方 Binjiang_Competition 安装指南 |
| 正式 Pi05 LoRA | Linux NVIDIA GPU；官方估计显存 >22.5 GB | 固定提交 OpenPI 的 `uv.lock` |

不要将本仓库 Windows `requirement.txt` 或开发锁安装进 OpenPI 环境。OpenPI 工具直接从源码脚本运行，不需要安装本项目的 OrcaGym 包依赖。当前只有 Windows CPU 调试训练通过，正式 Linux 微调还未实测。

## 1. 离线开发与自检

```text
python -m venv .venv
```

Windows 激活 `.venv\Scripts\Activate.ps1`，Linux 执行 `source .venv/bin/activate`，然后：

```text
python -m pip install -r requirements-dev.txt
python -m pip install --no-deps -e .
python -m pip check
ruff check src scripts tests
ruff format --check src scripts tests
python -m unittest discover -s tests -v
python -m build --no-isolation
```

此流程不启动仿真，不下载模型权重。源码工具在 `scripts/`，可安装观测模块在 `src/unitree_vision/`。从 Git checkout 使用采集/训练脚本；wheel 主要提供观测命令。

## 2. 携带已有 60 条示范

数据被 Git 忽略，单独克隆代码不会获得视频、动作或训练统计。旧电脑执行：

```text
python scripts/export_training_bundle.py data/exports/g1_batch60.zip
```

将 ZIP 和同名 `.sha256` 文件复制到新电脑。ZIP 包含首批 60 条有效示范及失败审计记录、训练/验证清单、新训练统计与准备报告；不含大模型权重、OrcaLab 资产、旧的三条试验数据或私人凭据。包内旧日志的绝对路径只用于历史追溯，训练清单使用相对路径。

先校验再解压到项目根目录，不要覆盖另一批同名数据：

```text
python scripts/export_training_bundle.py /path/to/g1_batch60.zip --verify
python -m zipfile -e /path/to/g1_batch60.zip .
python scripts/verify_batch.py data/batch_20261005
```

目录应为 `项目/data/batch_20261005/manifest_train.json`。训练 48 条、验证 12 条，按完整 episode 分离。`configs/multitask_g1.json` 仅引用历史三条试验数据；新训练使用本批次清单。

## 3. 在 Linux 准备 OpenPI

先安装 Git、Git LFS、uv 和适配 GPU 的 NVIDIA 驱动；确认 `nvidia-smi`。在本仓库相邻位置克隆官方源码：

```bash
git clone https://github.com/Physical-Intelligence/openpi.git ../openpi
git -C ../openpi checkout 981483dca0fd9acba698fea00aa6e52d56a66c58
git -C ../openpi submodule update --init --recursive
cd ../openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
cd ../unitree-g1-auto-recognition
```

下列命令使用 OpenPI 自己的 Python，不安装本项目的开发锁。若项目目录名称不同，请调整上一条 `cd`。先确认 GPU，再检查数据：

```bash
../openpi/.venv/bin/python -c "import jax; print(jax.devices())"
../openpi/.venv/bin/python scripts/verify_batch.py data/batch_20261005
../openpi/.venv/bin/python scripts/prepare_multitask_training.py \
  --openpi-root ../openpi \
  --train-manifest data/batch_20261005/manifest_train.json \
  --validation-manifest data/batch_20261005/manifest_validation.json \
  --output data/training_prepared_new
```

准备目录必须不存在；统计只使用训练集。正式训练前先不加 `--train` 检查配置：

```bash
../openpi/.venv/bin/python scripts/train_g1_openpi.py \
  --openpi-root ../openpi \
  --dataset data/batch_20261005/manifest_train.json \
  --assets data/training_prepared_new/assets \
  --checkpoints data/checkpoints --experiment g1_multitask_v1 \
  --batch-size 1 --steps 1000
```

确认列出 GPU 后，加 `--train` 才会下载官方 Pi05 权重并运行训练。首次建议新实验名配 `--steps 1`，完成前向、反向、优化器更新和检查点保存的显存试验，再换新实验名跑长训练。脚本尚无 resume 参数；不要把重用实验名当成恢复训练。

当前训练入口尚未自动运行验证集评估，正式 Linux 链路、显存和检查点都需实测。CPU `check_openpi_training.py` 使用随机初始化 dummy 网络，只作诊断，不产出可部署策略。详情见 [训练状态](multitask_training_batch60.md)。

## 4. 新环境继续仿真采集

训练服务器不需要安装 OrcaLab。若也要采集，在另一独立环境按官方 Windows 指南安装依赖，并把官方仓库放在相邻目录：

```text
git clone https://github.com/openverse-orca/Binjiang_Competition.git ../Binjiang_Competition
git -C ../Binjiang_Competition checkout b7ab885758030dce75f6fe71ac61b97775b292d6
git -C ../Binjiang_Competition apply --check ../unitree-g1-auto-recognition/docs/official_windows_24fps.patch
git -C ../Binjiang_Competition apply ../unitree-g1-auto-recognition/docs/official_windows_24fps.patch
```

官方仓库使用其 README 和环境配置安装 LeRobot/third_party 等运行依赖；不要用本项目开发锁代替。补丁必须应用在固定版本上，已应用时不要重复应用。订阅官方场景与机器人资产，打开官方 `g1_pick_buttons.json`，把头部和右腕实际相机设置为 1280×960，启用 Color Camera/NVENC，端口为 7090/7080。用“无仿真程序（手动启动）”进入 Runtime；端口连通不代表已进入 Runtime。

激活该采集环境，在本项目根目录运行：

```text
python scripts/batch_collect.py --output data/new_batch --per-task 20 --max-new 3
python scripts/batch_collect.py --output data/new_batch --resume
```

默认从相邻 `Binjiang_Competition` 读取官方脚本；可通过 `G1_OFFICIAL_ROOT` 指定别的位置。每类最多尝试 25 条，连续三次失败停止。批次内创建 `STOP` 文件可在当前示范结束后停止。不要并行运行两个控制器。

单条采集 PowerShell 入口可用 `-Python python` 或传入解释器完整路径，`-OfficialRoot` 可更改官方仓库位置。更多操作见 [批量采集](batch_collection.md)、[官方采集](official_demo_24fps.md)。

## 能力与数据边界

- 原始观测模块：45/30/15 维实测关节状态，按帧号配对，`action=null`。
- 官方动作示范：18 维双臂末端位姿（基座坐标、xyzw）和夹爪控制值；`action[t]=state[t+1]`，不可再偏移；相机采用最近帧采样，尚非严格物理同帧。
- 目标解析支持文字/人工参考框，尚无通用检测/OCR。开发成功判定使用场景关节，不是官方裁判。
- 本批只变化接近路点和动作时长；柜体和光照不变，不能证明跨场景泛化。没有可直接下发给机器人的正式训练策略。
