# Unitree G1 自主视觉观测工程

为 OrcaLab 26.8.2 的宇树 G1 提供相机观测、关节状态和严格按帧号配对的数据采集。当前版本 **0.1.0** 是后续识别与 OpenPI 策略开发的基础，尚未实现目标识别、策略训练、动作执行或成功判断。

头部与右腕图像通过 H.264 WebSocket 接收；状态来自控制程序的本地 MuJoCo；图像和状态使用相同 `simulate_index`。无法配对的帧会跳过并计数，不使用最近一帧替代。

## 安装与启动

Windows 11 x64，已有 `orcalab` Conda 环境，Python 3.12。完整 GPU/Conda 配置见 [Windows 环境说明](docs/windows_setup.md)。

```powershell
conda activate orcalab
# 包含 OrcaLab、视觉开发工具及固定的 Windows 依赖
python -m pip install -r requirement.txt
python -m pip install --no-deps -e .
python -m pip check
```

已有环境安装好依赖时，只需 `python -m pip install --no-deps -e .`。CPU/cu128 安装仍可使用 `scripts/setup_windows.ps1`；脚本会同时安装本工程。

在 OrcaLab 中切换到 `binjiang_competition_2026`，打开官方 `g1_pick_buttons.json` 布局，启动时选择“无仿真程序（手动启动）”。检查头部/右腕 Color Camera 和 UseNvEnc 已启用，端口分别为 **7090 / 7080**。保持你已保存的 1280×960 布局；程序不会擅自修改相机分辨率，保存图片保留原始尺寸。

```powershell
# 双路预览：S 保存；R 切换连续记录；Q/Esc 退出
python scripts/observation_collector.py --config configs/collector.toml
# 也可使用安装后的命令
g1-observe --config configs/collector.toml --prompt "识别面板上的按钮4"
# 5 秒无人值守试采集，含末尾等待配对阶段
g1-observe --config configs/collector.toml --headless --record --duration 5
```

**独立采集模式会加载 SDK 模型、初始化场景姿态并暂停服务端物理；不推进物理，也不适合和机器人控制程序同时运行。** 它用于固定姿态观测。运动过程采集应嵌入已有控制程序，详见 [操作与联调](docs/operations.md)。

配置使用 TOML `[collector]` 表，CLI 参数覆盖配置；错误类型、重复相机名和无效数值会提前报错。相机队列、待配对队列均有容量上限。

## 数据与诊断

每次采集创建独立运行目录：

```text
data/observations/<run>/
  manifest.json              # 状态顺序/单位、相机属性、实际配置、依赖版本
  index.jsonl                # 完整样本索引，可离线重建
  summary.json               # 保存/配对/丢帧/队列溢出计数与退出状态
  samples/<sample_id>/
    head.png
    wrist_r.png
    observation.json        # 指令、状态、仿真帧号、各图像元数据
```

```powershell
# 把 <run> 换成实际运行目录名
g1-dataset data/observations/<run>
# 采集停止后，校验完整样本并重建索引；原索引备份，不删除数据
g1-dataset data/observations/<run> --rebuild-index
# 原来的诊断命令继续可用
python scripts/camera_preview.py
python scripts/camera_preview.py --receive-only
python scripts/robot_state.py --profile right --duration 10
```

默认状态为 45 维关节位置，另存对应速度；`arms` 为 30 维，`right` 为 15 维。夹爪记录实际连杆角度，不代表开口宽度。`action` 和 `annotation` 为 `null`：目前是观测数据，不能直接用于模仿学习动作策略。详见 [数据契约](docs/data_contract.md)。

## 开发与迭代

```powershell
# 新建独立开发环境，避免覆盖 orcalab 的桌面依赖版本
conda create -n g1_dev python=3.12
conda activate g1_dev
# 核心采集依赖，不安装 Torch、OrcaLab 桌面或模型权重
python -m pip install -r requirements-dev.txt
python -m pip install --no-deps -e .
ruff check src scripts tests
ruff format --check src scripts tests
python -m unittest discover -s tests -v
python -m build --no-isolation
```

GitHub Actions 在 Windows 和 Linux 的 Python 3.12 上运行离线测试、格式检查与构建。硬件取流须按 [联调验收](docs/operations.md) 单独验证。

- [模块边界与 OpenPI 路线](docs/architecture.md)：观测 → 任务理解 → 策略 → 执行 → 结果判断。
- [数据字段、同步与恢复](docs/data_contract.md)。
- [贡献与版本演进](CONTRIBUTING.md)、[变更记录](CHANGELOG.md)。
- [代码来源和验证记录](docs/validation.md)。

采集数据、权重、运行产物不提交 Git。训练环境与采集环境分开；本工程不安装 OpenPI 训练端，也不假设某个预训练模型已适配 G1。
