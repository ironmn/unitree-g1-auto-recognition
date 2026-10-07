# Linux 多任务 OpenPI 微调指导手册

更新：2026-10-07。适用于本项目提交 `2bf051921dc96e759b473bc254871aa3dc9bf955` 与 OpenPI 提交 `981483dca0fd9acba698fea00aa6e52d56a66c58`。

**状态：已在 WSL Ubuntu 26.04 + RTX 5080 上完成真实 Pi05 预训练模型的 100 步微调，并完成训练前后模型的独立离线评估。** 原配置在反向传播时显存不足；成功配置额外冻结视觉编码器，并优化主机内存，详见第十三节。离线评估使用 12 条留出示范，完整指标和工件核验通过；模型误差仍大于保持当前状态的参照。Windows OrcaLab 模型闭环操作及成功率尚待测试，不能用离线结果代替。[离线评估操作说明](offline_evaluation.md)与[本次实测结果](evaluation_results_20261007.md)记录详细证据。

前十二节的基线提交为 `2bf0519`；第十三节的新选项来自本次本地代码扩展，旧基线没有这些选项。迁移时使用 `data/exports/g1_wsl_training_code_20261007.zip` 内的完整源码，或包含这些新增文件的工作目录。

该源码 ZIP 是训练完成时的快照，早于新增离线评估入口。要复现离线评估，另需包含 `scripts/evaluate_g1_openpi.py` 与 `scripts/g1_openpi_eval.py` 的当前仓库源码；不要将旧训练源码包当作完整评估代码包。

目标是使用官方预训练 Pi05，学习“按压停止按钮、旋转旋钮、拨动拨杆”三类示范。方法为双分支 LoRA，批大小 1，24 步动作窗口。不是重新训练一个大模型，也不是运行随机初始化的 CPU 调试网络。

## 一、开始前准备

| 项目 | 要求与说明 |
|---|---|
| 系统 | 以 Ubuntu 22.04 为基准；这是固定版本 OpenPI 文档测试的系统 |
| GPU | NVIDIA，官方文档 LoRA 显存估计 **>22.5 GB**；24 GB 级别仍应先实测峰值，不能保证任意配置都能运行 |
| 驱动 | `nvidia-smi` 正常，且 JAX 能枚举 GPU；仅前者通过不够 |
| 内存 | 当前适配器将所有图像缩到 224×224 后放入 RAM，48 条训练集图像约占 4.9 GB，另需模型初始化、数据和编译内存 |
| 磁盘 | 数据 ZIP 约 491 MB；解压、Python/CUDA 依赖、预训练权重及多个检查点另占空间，不能只按 ZIP 大小分配 |
| 网络 | 能访问 GitHub、Python 软件源和 Google Cloud Storage 的官方模型/分词器资源 |
| 数据 | `g1_batch60_20261006.zip` 及同名 `.sha256`，Git 仓库不含数据 |

训练服务器不需要安装 OrcaLab 或订阅仿真资产。不要向 OpenPI 环境安装本仓库的 Windows `requirement.txt` 或 `requirements-dev.txt`，它们属于另一条环境链路。

资源要求依据固定版本的 [OpenPI README](https://github.com/Physical-Intelligence/openpi/blob/981483dca0fd9acba698fea00aa6e52d56a66c58/README.md)。第十三节提供本机 16 GB 显卡已验证的冻结视觉编码器配置；它与前十二节的原配置训练范围不同。

## 二、安装基础工具并建立目录

以下命令均为 **Linux Bash**。已有的工具无需重复安装；托管服务器若无 sudo 权限，使用管理员已经提供的 Git、uv、tmux。

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs curl tmux build-essential pkg-config libgl1 libglib2.0-0
git lfs install
nvidia-smi
free -h
df -h
```

安装 uv（来源：[uv 官方安装说明](https://docs.astral.sh/uv/getting-started/installation/)）：

```bash
curl -LsSf https://astral.sh/uv/install.sh -o /tmp/g1-uv-install.sh
sh /tmp/g1-uv-install.sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
mkdir -p ~/g1-work
cd ~/g1-work
```

如果 `nvidia-smi` 失败，先让服务器提供方修复驱动或容器 GPU 透传，不要继续用 CPU 安装结果冒充 GPU 环境。

## 三、获取固定版本代码

```bash
cd ~/g1-work
git clone https://github.com/ironmn/unitree-g1-auto-recognition.git
git -C unitree-g1-auto-recognition checkout 2bf051921dc96e759b473bc254871aa3dc9bf955

GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Physical-Intelligence/openpi.git
git -C openpi checkout 981483dca0fd9acba698fea00aa6e52d56a66c58
git -C openpi submodule update --init --recursive
```

固定本项目提交可避免依赖 PR 是否合并；checkout 后处于 detached HEAD，适合复现。若要修改代码，另建分支。两个目录应为：

```text
~/g1-work/
├── openpi/
└── unitree-g1-auto-recognition/
```

## 四、安装 OpenPI 独立环境

```bash
cd ~/g1-work/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --python 3.11
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
uv pip check
.venv/bin/python -c "import jax; print('JAX:',jax.__version__); print('devices:',jax.devices()); assert any(d.platform=='gpu' for d in jax.devices()), 'No JAX GPU'"
```

**验收：最后一条必须成功且列出 GPU。** 不应只出现 `CpuDevice`。请以 OpenPI 锁定依赖为准，不直接升级 JAX、Flax、NumPy 或 CUDA wheel；如果新 GPU 与锁定版本不兼容，需要单独处理并记录改动。

之后统一使用 `../openpi/.venv/bin/python`，无需激活原有 Conda 环境。新 SSH 会话也可以使用这个解释器路径。

本次使用 Python 3.11.17。固定依赖中的 MuJoCo 2.3.7 在 Python 3.12 Linux 上转为源码构建并失败，而 3.11 可以使用 wheel。这个训练环境与本项目 Python 3.12 的生产采集/开发环境独立。

## 五、传输、校验并解压示范数据

旧 Windows 电脑上的文件位于项目 `data/exports/`：

```text
g1_batch60_20261006.zip
g1_batch60_20261006.zip.sha256
```

可用 SFTP 工具上传，或在 Windows PowerShell 执行以下示例；将服务器用户名、地址替换为真实值，并先在服务器创建目标目录。**这两行在 Windows 执行，后续回到 Linux。**

```powershell
scp .\data\exports\g1_batch60_20261006.zip your_user@your_server:~/g1-work/
scp .\data\exports\g1_batch60_20261006.zip.sha256 your_user@your_server:~/g1-work/
```

Linux 中校验外层 ZIP 和包内全部文件：

```bash
cd ~/g1-work
sha256sum -c g1_batch60_20261006.zip.sha256
cd unitree-g1-auto-recognition
../openpi/.venv/bin/python scripts/export_training_bundle.py ../g1_batch60_20261006.zip --verify
```

两项校验通过后再解压。若已经有 `data/batch_20261005` 或 `data/multitask_training_20261006`，先确认不是另一份工作成果，不要直接覆盖。

```bash
../openpi/.venv/bin/python -m zipfile -e ../g1_batch60_20261006.zip .
../openpi/.venv/bin/python scripts/verify_batch.py data/batch_20261005
```

**验收：`passed: true`，60 条、20,301 帧，训练 48 条、验证 12 条。** 校验会抽取每个任务、每个划分各一条真实示范，经过 OpenPI 输入适配器。

包内历史日志的旧 Windows 路径只是追溯信息；实际数据清单使用相对路径，不要批量替换日志或重新写审计哈希。

## 六、重新生成训练集统计

```bash
cd ~/g1-work/unitree-g1-auto-recognition
mkdir -p data/linux_runs/logs
../openpi/.venv/bin/python scripts/prepare_multitask_training.py \
  --openpi-root ../openpi \
  --train-manifest data/batch_20261005/manifest_train.json \
  --validation-manifest data/batch_20261005/manifest_validation.json \
  --output data/linux_runs/prepared_v1
```

**验收：** `prepared_not_trained`；训练 48 条、16,323 帧；验证 12 条未参与统计；`statistics_roundtrip_passed: true`。统计文件应在：

```text
data/linux_runs/prepared_v1/assets/g1_buttons/norm_stats.json
```

输出目录必须是新目录。已有统计可继续使用，不要为了重新运行命令删除原目录；修改数据时使用新的准备目录，并记录对应清单。

## 七、检查训练配置（尚不训练）

```bash
../openpi/.venv/bin/python scripts/train_g1_openpi.py \
  --openpi-root ../openpi \
  --dataset data/batch_20261005/manifest_train.json \
  --assets data/linux_runs/prepared_v1/assets \
  --checkpoints data/linux_runs/checkpoints \
  --experiment g1_smoke_v1 --batch-size 1 --steps 1
```

**验收：** `mode: inspect_only`，GPU 设备，机器人维数 18、模型维数 32、窗口 24。此步骤只构造配置，不证明权重可以加载，也不证明显存足够。

模型权重固定为 `gs://openpi-assets/checkpoints/pi05_base/params`，默认缓存到 `~/.cache/openpi`。可在启动前更改缓存位置：

```bash
export OPENPI_DATA_HOME="$HOME/g1-work/model-cache"
mkdir -p "$OPENPI_DATA_HOME"
```

## 八、先做一次完整训练步

为防止 SSH 断开终止训练，先运行 `tmux new -s g1-train`。进入会话后重新执行下面的路径与环境设置。

```bash
cd ~/g1-work/unitree-g1-auto-recognition
export OPENPI_DATA_HOME="$HOME/g1-work/model-cache"
export PYTHONUNBUFFERED=1
set -o pipefail

../openpi/.venv/bin/python scripts/train_g1_openpi.py \
  --openpi-root ../openpi \
  --dataset data/batch_20261005/manifest_train.json \
  --assets data/linux_runs/prepared_v1/assets \
  --checkpoints data/linux_runs/checkpoints \
  --experiment g1_smoke_v1 --batch-size 1 --steps 1 --train \
  2>&1 | tee data/linux_runs/logs/g1_smoke_v1.log
```

另一个终端监视 `watch -n 1 nvidia-smi`。首次下载权重、解码数据和 JAX 编译可能较慢，不能仅凭几分钟没有新 step 就判断卡死。

**验收必须同时满足：**

1. 权重加载成功，无缺失参数或形状不匹配异常。
2. 完成前向、反向和优化器更新，损失与梯度日志没有 NaN/Inf。
3. 程序正常退出，最后的检查点异步写入完成。
4. 检查点目录包含 `params`、`train_state` 和 `assets`；只跑一轮时目录编号可能是 `0`，因为官方训练循环从 0 计数。

检查产物：

```bash
find data/linux_runs/checkpoints/pi05_g1_multitask_lora/g1_smoke_v1 -maxdepth 3 -type d
tail -n 40 data/linux_runs/logs/g1_smoke_v1.log
```

`pipefail` 可以避免训练失败却被 `tee` 的成功退出码掩盖。出现 OOM 后不要继续长训练；批大小已经是 1，梯度累积并不能解决单个样本都放不下的问题。官方的 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` 仅调整预分配比例，不增加总显存；只有确认独占 GPU 且留有系统余量时才考虑设置。

## 九、运行首轮 1,000 步微调

单步成功后，在同一 tmux 会话中执行。**使用新的实验名**，此任务从官方基础权重开始，不会自动接着单步检查点训练。

```bash
cd ~/g1-work/unitree-g1-auto-recognition
set -o pipefail
../openpi/.venv/bin/python scripts/train_g1_openpi.py \
  --openpi-root ../openpi \
  --dataset data/batch_20261005/manifest_train.json \
  --assets data/linux_runs/prepared_v1/assets \
  --checkpoints data/linux_runs/checkpoints \
  --experiment g1_multitask_1000_v1 --batch-size 1 --steps 1000 --train \
  2>&1 | tee data/linux_runs/logs/g1_multitask_1000_v1.log
```

当前入口每 10 步输出训练日志、每 100 步及最后一步触发检查点保存；官方保留策略可能清理部分中间检查点，因此不保证每个历史步骤目录都存在。`1,000 步` 是首次试验规模，不是保证收敛的训练处方；步数也不等于完整遍历数据集的次数。

tmux 按 `Ctrl+B` 再按 `D` 可退出显示但保持任务运行；重新连接用 `tmux attach -t g1-train`。需要停止时在该会话中按 `Ctrl+C`，不要将写入中的检查点视为可用结果。

## 十、保存结果与正确评估

完成后保留整个实验目录、对应训练统计、日志和代码版本，不只复制 `params` 子目录。建议在每次试验开始时保存：

```bash
git rev-parse HEAD > data/linux_runs/logs/project_commit.txt
git -C ../openpi rev-parse HEAD > data/linux_runs/logs/openpi_commit.txt
uv pip freeze --python ../openpi/.venv/bin/python > data/linux_runs/logs/environment.txt
nvidia-smi > data/linux_runs/logs/gpu.txt
```

训练完成后分三层验收：

| 层次 | 要验证的内容 | 当前支持情况 |
|---|---|---|
| 训练计算 | 实际梯度更新、有限损失、可恢复检查点 | WSL 冻结视觉编码器配置已验证 100 步；原配置显存不足 |
| 独立离线评估 | 固定验证噪声/采样设置，按三任务报告验证损失，始终使用训练统计 | `scripts/evaluate_g1_openpi.py` 已完成 12 条留出示范的训练前后配对评估，结果核验通过 |
| 仿真闭环 | 当前图像/状态/指令→短段动作→重新观测，统计成功率、误操作、超时等 | **尚未实现正式模型闭环验证** |

不能把训练损失下降等同于机器人学会操作。验证集清单是 `data/batch_20261005/manifest_validation.json`，不能加入训练或重算归一化。评估入口、命令和统计方法见[独立离线评估](offline_evaluation.md)。本次宏平均验证 loss 为 `0.180635 → 0.167229`，双臂位置误差为 `57.408 → 55.042 mm`；右臂仍有 `105.924 mm` 的完整窗口位置误差，保持当前状态参照仅 `41.870 mm`。当前改善有限，尚不能证明模型能完成操作。

部署前还需要恢复正确模型配置、反归一化输出，处理四元数归一化、动作范围、时序与控制器衔接。当前 18 维动作是末端位姿/夹爪控制表示，不能直接当成 18 个电机角度下发。

## 十一、常见问题

| 现象 | 处理 |
|---|---|
| `nvidia-smi` 成功，JAX 只有 CPU | 核对使用的是 `openpi/.venv/bin/python`，检查 JAX/CUDA 依赖与驱动；先解决 GPU 枚举 |
| `Expected OpenPI 981483d` | 官方代码提交不匹配，回到第三节固定版本，不删掉脚本保护 |
| `No such file` 指向数据 | 确保 ZIP 解压到项目根目录，保留 `data/batch_20261005/` 层级；不要使用历史三条数据的 `configs/multitask_g1.json` |
| 准备目录已存在 | 使用已有结果或新的目录名；脚本有意防止覆盖 |
| 实验目录已存在 | 换新实验名；当前包装器**没有 `--resume` / `--overwrite` 参数**，重复运行不是续训 |
| 权重/分词器下载失败 | 检查访问 Google Storage、代理和缓存盘空间，保留原错误；不要改用随机权重继续 |
| 第一次反向传播 OOM | 查看其他进程及显存峰值；batch 已为 1，需更大显存或单独验证低显存实现 |
| 进程只显示 `Killed` | 检查主机/容器 RAM 限制和系统 OOM 记录，不一定是 GPU 显存 |
| 日志几分钟不更新 | 分辨正在下载、视频解码还是首次 JAX 编译，观察 CPU/GPU/磁盘；避免重复启动第二个训练进程 |
| 训练结束却没成功率 | 离线评估已有自动入口，但操作成功率需要新的仿真闭环测试；尚未测量不等于 0 或 100% |

如果需要中断恢复，应先为包装器接入并测试官方的 resume 机制，再进行长周期训练；不要手工移动或拼接检查点冒充恢复成功。

## 十二、本批数据能证明什么

目前三类各 20 条，共 60 条有效示范，48/12 条分离。它们只随机改变初始接近路点和动作时长，未随机化柜体位置、光照或相机视角。成功筛选使用开发版场景关节判据，未接入官方裁判；官方相机最新帧采样也尚非严格物理同帧。

首轮训练链路和独立离线评估基线均已建立。本次留出示范只衡量同场景轨迹变化；跨场景识别和操作泛化需要后续增加变化数据并测试，不能靠本批训练或验证损失推断。

## 十三、本地 WSL 16 GB 显卡的已验证配置

实测环境为 Ubuntu 26.04.1、Python 3.11.17、JAX 0.5.3、Flax 0.10.2、RTX 5080（NVIDIA-SMI 总显存 16,303 MiB）。使用全部 48 条训练示范，24 FPS，24 步窗口，batch 1，无 EMA。OpenPI 上游源码保持固定提交，没有修改上游文件。

原配置完成了权重加载和 GPU 初始化，但第一步反向传播申请额外约 5.37 GiB 时失败。成功配置使用以下三个新增选项：

| 选项 | 实际作用 |
|---|---|
| `--image-cache` | 按视频内容和解码/缩放版本生成只读 RGB 内存映射；像素与原解码一致，避免把约 4.9 GB 图像永久放入匿名 RAM |
| `--memory-efficient-restore` | 按参考训练数据类型逐个恢复大权重；与原转换逐值一致，避免旧 Orbax 的 NumPy 并行读取耗尽 RAM |
| `--freeze-vision` | 冻结视觉编码器并按官方训练器规则转为 bfloat16，保留双 LoRA、动作投影和时间 MLP 的训练；这会改变可训练参数范围 |

模型仍使用完整 Pi05 预训练网络。最后一项是训练配置变更，不能将这次结果描述为原配置全部参数范围的微调成功。

迁移到新环境时，先按第二至五节建立目录、安装依赖和导入数据，再将本次源码 ZIP 与 `.sha256` 放到 `~/g1-work/`，校验后覆盖基线源码：

```bash
cd ~/g1-work
sha256sum -c g1_wsl_training_code_20261007.zip.sha256
openpi/.venv/bin/python -m zipfile -e g1_wsl_training_code_20261007.zip unitree-g1-auto-recognition
```

源码包记录的是本次含未提交修改的文件快照，逐文件哈希位于 `SOURCE_MANIFEST.json`。它不会替代原始数据包、模型缓存或 OpenPI 环境。

在本机 Linux Bash 中复现以下命令。实验名使用新名字，从官方基础权重开始；不会自动接着已有检查点续训。

```bash
cd ~/g1-wsl-20261007/project
export OPENPI_DATA_HOME="$HOME/g1-wsl-20261007/model-cache"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
set -o pipefail
../openpi/.venv/bin/python scripts/train_g1_openpi.py \
  --openpi-root ../openpi \
  --dataset data/batch_20261005/manifest_train.json \
  --assets data/wsl_finetune_20261007/prepared/assets \
  --checkpoints data/wsl_finetune_20261007/checkpoints \
  --image-cache data/wsl_finetune_20261007/image_cache \
  --memory-efficient-restore --freeze-vision \
  --experiment g1_multitask_frozen_vision_100_v2 \
  --batch-size 1 --steps 100 --train \
  2>&1 | tee data/wsl_finetune_20261007/logs/g1_multitask_frozen_vision_100_v2.log
```

迁移到另一台 Linux 时，将 `cd`、OpenPI 根目录、数据、统计和缓存路径改成对应的位置。依赖仍按第四节安装，原始数据仍按第五节转移；源码包不含数据、依赖、权重或训练日志。

实测 100 次优化器更新已完成，保存目录编号为 `99`，恢复后的 `train_state.step` 为 `100`。全部 20 个 LoRA 参数张量相对于同一 GPU、同一 seed 42 的上游初始化值发生了变化；上游 A/B 均为正态随机初始化，不能以“非零”作为训练证明。最终优化器数组和已记录指标均有限，检查点训练统计与准备阶段一致。整卡占用峰值 11,321 MiB（包括桌面等占用），进程 RAM 峰值约 13.0 GiB，交换内存峰值约 3.88 GiB；恢复、编译、100 步训练和保存合计约 150.5 秒，首次安装及下载另计。

这仍是短程实验：实际训练沿用上游 1,000 步学习率预热，100 步全部位于预热阶段，尚不能按充分训练的策略解读。主机内存余量较紧，保存时明显依赖交换空间；不能据此保证更大 batch、其他相机配置或更长训练一定稳定。独立离线评估已完成并显示有限改善，下一步应验证仿真闭环操作，再据失败原因调整数据、控制和训练规模；详见[实测结果](evaluation_results_20261007.md)。
