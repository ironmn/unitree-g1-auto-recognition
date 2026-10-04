# unitree-g1-auto-recognition

虚拟环境中的宇树机器人自主识别

# 宇树 G1 · 任务二视觉识别环境

Windows 11 x64、Miniconda、已有 `orcalab` 环境、Python **3.12**。本仓库准备相机取流、OpenCV 图像处理、YOLO 检测和 OrcaGym 接口依赖；已集成相机预览、状态读取和同步观测采集，尚未实现视觉识别算法或动作控制。

任务目标来自比赛手册 §4.2：识别旋钮 1、拨杆 3，以及按钮 2/4/5/6/7 中的目标位置。识别得到的二维像素位置仍需相机标定、深度或面板几何约束、坐标变换，才能交给末端控制器。通用 YOLO 权重不能直接区分这些比赛编号，需要场景数据标注和训练或专门的几何识别规则。

## 1. 修复 PowerShell 的 Conda 提示符

从开始菜单打开 **Miniconda Prompt**，执行：

```cmd
conda env list
conda init powershell
conda config --set changeps1 true
conda config --set env_prompt "({name}) "
```

关闭现有 PowerShell 标签页，在 Windows Terminal 中新开 Windows PowerShell，然后执行：

```powershell
conda activate orcalab
$env:CONDA_DEFAULT_ENV
python -c "import sys; print(sys.executable); print(sys.version)"
```

预期提示符为 `(orcalab) PS D:\...>`，环境变量为 `orcalab`，Python 路径位于对应环境内。环境名字与 Python 路径是判断激活的依据。

若 PowerShell 报“禁止运行脚本”，先查看 `Get-ExecutionPolicy -List`。个人电脑可执行 `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` 后重开终端；组织的 GroupPolicy 限制需要遵从管理员配置。不要只为显示名字手工覆盖 `prompt`。

若 `conda` 在 PowerShell 中仍找不到，返回 Miniconda Prompt 执行 `where conda`，用查到的 `condabin\conda.bat` 完整路径初始化：

```powershell
# 替换为 where conda 返回的真实路径
& 'C:\实际安装目录\Miniconda3\condabin\conda.bat' init powershell
```

重开终端。如果激活成功但没有括号，检查 `$PROFILE` 是否加载 Conda 初始化块、终端是否以 `-NoProfile` 启动，以及 Oh My Posh/Starship 等自定义提示符是否隐藏 Conda 名字。Windows PowerShell 与 PowerShell 7 的配置文件可以不同。

## 2. 安装到已有环境

在本仓库根目录运行。先确认 `conda list -n orcalab python` 的版本为 3.12；脚本遇到其他版本会停止，不会删除或重建环境。若需要变更 Python，请先确认已有项目兼容后再执行 `conda install -n orcalab python=3.12`。

CPU 安装适合先检查环境；不会使用 GPU 加速 YOLO：

```powershell
.\scripts\setup_windows.ps1 -TorchBackend cpu
```

有 NVIDIA 显卡、需要 GPU 推理时，先查看 `nvidia-smi` 并更新到支持 CUDA 12.8 的稳定驱动，再选择：

```powershell
.\scripts\setup_windows.ps1 -TorchBackend cu128
```

脚本通过官方 PyTorch 索引安装匹配的 Torch 2.8 / Torchvision 0.23，再安装 `requirement.txt`，执行 `pip check` 和离线验证。RTX 50 系列应使用支持其架构的 CUDA 构建；本方案提供 cu128。通常无需为 PyTorch wheel 单独安装 CUDA Toolkit。

如脚本被执行策略阻止，可在当前终端使用 `Set-ExecutionPolicy -Scope Process Bypass` 后重试（仅对这个进程生效，不覆盖组织策略），或手动运行：

```powershell
conda activate orcalab
python -m pip install --upgrade pip
# CPU；GPU 方案把索引最后的 cpu 改为 cu128
python -m pip install --upgrade torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirement.txt
python -m pip check
python scripts/verify_environment.py
# GPU 方案追加验证：
# python scripts/verify_environment.py --require-cuda
```

依赖下载失败时可给安装 requirement.txt 的命令加 `-i https://pypi.tuna.tsinghua.edu.cn/simple`；PyTorch 的 CPU/CUDA 专用索引保持原样。不要使用 `--no-deps` 安装本清单，也不要同时安装多个 OpenCV 发行包（如 headless/contrib），它们共享 `cv2` 命名空间。

安装成功后可保存本机实际版本：

```powershell
python -m pip freeze > requirements-local.txt
conda env export -n orcalab > environment-local.yml
```

`requirement.txt` 固定直接与间接依赖；`requirements.in` 记录选型。Torch 的 CPU/cu128 本地版本由安装选项决定，`==2.8.0` 接受对应的 `+cpu` / `+cu128` 构建。本清单只面向 Windows Python 3.12，不用于替换官方完整 LeRobot 采集环境，也不是跨平台锁文件。

## 3. 启动与相机联调

```powershell
conda activate orcalab
orcalab
```

首次启动还可能安装平台内部依赖。订阅 `Binjiang_Competition_2026`、`g1_pick`，重启 OrcaLab 同步资产。按官方代码包加载 `src/examples/dataCollection/unitree_g1/g1_pick_buttons.json`，检查 Color Camera、UseNvEnc 已启用，头部端口 7090、右腕 7080，启动仿真等待 gRPC 50051 就绪。

相机视频通过 WebSocket 传输，包含协议帧头，不能直接把端口当作 `cv2.VideoCapture` 视频源。复用官方 OrcaGym 相机接口或官方取流代码，保持 `env.render()` 调用才会持续产帧。实际相机连接、NVENC 编码、平台启动和控制动作必须在本机联调；离线验证通过不代表这些链路已经通过。

若后续采用官方 LeRobot 采集工具，按官方 Windows 指南另建 `orcalab_lerobot` 环境，并安装官方仓库内三个 third_party 源码包。本仓库不混装该训练/采集工具链。新的采集工程安装步骤见根目录 README。

## 依据与验证范围

- 比赛手册：Python 3.12，OrcaLab/OrcaGym 同系列且 ≥26.8.2。
- [官方仓库 Windows 安装说明](https://github.com/openverse-orca/Binjiang_Competition#安装运行环境)。
- [官方 Windows 相机与采集指南](https://github.com/openverse-orca/Binjiang_Competition/blob/main/docs/unitree_g1_collection_windows.md)。
- 参考源码提交：`b7ab885758030dce75f6fe71ac61b97775b292d6`。
- 已在 Linux 工作区用 uv 对 Windows x64 / Python 3.12 做仅 wheel 的依赖解析；未在用户 Windows 机器实际安装或验证。
- 自检覆盖导入、OpenCV 图像与几何运算、H.264 解码器、Torchvision NMS；选择 GPU 时增加实际 CUDA 运算。CPU 自检不要求 NVIDIA 显卡；运行官方仿真/采集还受其硬件要求约束。
