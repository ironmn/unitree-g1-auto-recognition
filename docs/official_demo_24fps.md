# 官方 G1 示范采集与回放验证

2026-10-05 已在本机 OrcaLab 26.8.2 中执行官方 `my_waypoint_press_01.yaml`（指令：按压停止按钮），完成试跑、正式数据保存和回放。

## 结果

| 项目 | 结果 |
|---|---|
| LeRobot 版本/格式 | 0.3.4+orca.1 / v2.1 |
| episode | 1 |
| 保存的状态/动作条数 | 275 |
| state/action | 各 18 维；action[t] = state[t+1] |
| 仿真采样与视频帧率 | 24 FPS |
| 视频 | 头部、右腕各 275 帧，1280×960 |
| 视频时间长度 | 11.4583 秒 |
| 官方轨迹执行 | 2300 个控制步，仿真时间约 11.5 秒 |
| 官方动作回放 | 2303 个控制步（含 10 步初始驻留与结束处理） |
| 数据验证 | 19/19 通过 |

帧数为 275 是因为原始采样的 276 个状态构成了 275 对相邻状态/动作，并非少保存了一帧。使用 `--clock sim`；真实墙钟耗时受渲染、网络与控制计算影响，不声称墙钟采样恰好每秒 24 次。

成功数据位于仓库内：

```text
data/official_demo_24fps/
  dataset_verified/       # 唯一完成提交并通过验证的数据集
    meta/
    data/chunk-000/episode_000000.parquet
    videos/chunk-000/observation.images.cam_head/episode_000000.mp4
    videos/chunk-000/observation.images.cam_wrist_r/episode_000000.mp4
  validation/validation.json
  validation/cam_head_contact_sheet.png
  validation/cam_wrist_r_contact_sheet.png
  capture_verified_audit/physics_summary.json
  capture_verified_audit/physics_trace.jsonl
  replay_audit/physics_summary.json
  replay_audit/physics_trace.jsonl
  runtime_versions.json
```

日志为 `data/official_capture_24fps_verified.log` 与 `data/official_replay_24fps.log`。首轮失败日志为 `data/official_capture_24fps.log`，首轮 `dataset/` 未完成提交，仅作排错材料，不能混入训练集。

验证包含：episode/帧数、连续索引、24 FPS 时间戳、18 维有限状态/动作、下一绝对状态标签、四元数范数、夹爪归一化范围、手臂/夹爪变化，以及完整解码两路视频核对帧数、帧率和分辨率。中文路径 JPEG 写入测试和实际回放类的 20/24 FPS 调度测试也通过。两路解码视频各有 275 个不同图像哈希。

物理审计检测到 `stop_button_joint` 在录制中约为 -2.000 至 +5.047 mm，回放约为 -2.000 至 +3.053 mm；这是关节坐标范围，不能解读为官方合格按压力或行程。其余五个按压式按钮关节未观察到变化；拨杆有约 0.077 rad 的漂移、旋钮约 0.0003 rad 的漂移，审计没有将其归因于误碰。该脚本使用 `EmptyTask`，任务结束主要表示轨迹执行到末尾，**没有读取官方裁判成功结果**。

官方采集模式读取最新相机图片；视频与 Parquet 帧数一致不证明它们与物理步严格同帧。训练前还应接入我们采集器的精确帧号配对。`joint_strip on / strip_col off` 是官方采集任务模型配置，不能据此宣称完整比赛碰撞条件下的性能。

## 运行环境和本地适配

官方源代码放在相邻的 `Binjiang_Competition/`，基准提交为 `b7ab885758030dce75f6fe71ac61b97775b292d6`。未推送 GitHub。

独立采集解释器：

```text
<collection-environment>/Scripts/python.exe
```

这是带 `system-site-packages` 的 venv：复用已有 `orcalab` 的 OrcaGym、PyAV、NumPy 和 CUDA Torch，新增 LeRobot/数据存储依赖安装在该 venv 内；未变更原有环境的软件包。实际版本记录在 `runtime_versions.json`，并非官方完整 CPU 锁定环境的逐项版本认证。

使用官方仓库随附的 `third_party/lerobot`、`televuer` 与 `openpi-client`。为 Windows 和 24 FPS 做了两项源代码适配，补丁保存于本目录的 `official_windows_24fps.patch`：

1. 官方统计 JPEG 写入由 `cv2.imwrite` 改为 `cv2.imencode` + `tofile`，解决中文 Windows 路径导致统计图像缺失、episode 保存失败。
2. 官方回放增加 `--replay_fps 24`，保留 5 ms 控制周期，以 8/9 个控制步调度相邻动作，避免原本固定 10 步把 24 FPS 数据按 20 FPS 播放。目标位姿和夹爪控制算法未改。

`run_official_demo.py` 调用官方 `main()` 并只读审计物理状态，不设置额外控制量。它在审计文件写完之后采用官方的进程退出方式，避免旧相机后台线程在解释器关闭时产生清理异常。已经提交的视频和统计数据不受首轮清理日志影响。

场景配置 `configs/official_g1_buttons.yaml` 对应已加载的场景名，保留灯光随机化为关闭；配置包含官方检查器要求的灯光字段。

## 复现命令

先启动 OrcaLab 的 `binjiang_competition_2026` 场景和 `g1_pick_buttons` 布局，以手动模式启动 Runtime，检查 gRPC `localhost:50051`、相机端口 7090/7080。不要同时运行另一个控制/采集程序；官方入口会重置机器人并执行动作。

在 `unitree-g1-auto-recognition` 根目录：

```powershell
# 重新采集一条示范；每次创建新的时间戳目录，不覆盖现有数据
.\scripts\run_official_demo_24fps.ps1 -Mode Collect

# 再次验证本次已保存数据
.\scripts\run_official_demo_24fps.ps1 -Mode Validate -RunDirectory .\data\official_demo_24fps

# 回放本次已保存数据；每次创建新的审计目录
.\scripts\run_official_demo_24fps.ps1 -Mode Replay -RunDirectory .\data\official_demo_24fps
```

验证模式完全离线。采集/回放模式只在仿真中执行官方示范，不代表自主策略推理。本次未训练模型，也未自动生成按钮位置标注。
