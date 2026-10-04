# 操作与联调

## 固定姿态 CLI

启动 OrcaLab 26.8.2，切换 `binjiang_competition_2026`，加载官方 G1 按钮布局；以“无仿真程序（手动启动）”进入 Runtime。相机启用 RGB、NVENC，头部/右腕端口 7090/7080。

独立采集会加载模型、初始化姿态并暂停服务端物理，结束时沿用原诊断工具的 RUNNING 恢复方式；不保存或恢复进入脚本前的完整机器人姿态。请勿与控制器同时运行。采集器不改变相机分辨率或 DDS 属性，不会自动下载模型权重。

```powershell
conda activate orcalab
g1-observe --config configs/collector.toml --headless --record --duration 5
```

持续采集使用 `--duration 0`。预览按 S 等待下一组完整样本，R 开关连续采集。CLI 和配置文件中的相对输出路径都相对当前工作目录。

启动会等待 WebSocket 连接，运行中等待完整配对。结束后还会等待最多 pair_timeout 秒补齐已提交样本，因而 duration 不是整个进程的严格时限。

## 接入已有控制器

在控制器所在的同一线程使用已经创建好的 env。此模式不新建、重置或推进物理环境；采集器只管理自身启用的相机流，结束时保留原来已开启的流和控制器环境。

```python
from pathlib import Path
from unitree_vision.config import CollectorConfig
from unitree_vision.collector import ObservationCollector

config = CollectorConfig(
    cameras="head,wrist_r",
    profile="arms",
    prompt="按下按钮4",
    record=True,
    output=Path("data/observations"),
)

# env 和 next_action 均由你的现有控制程序提供。
# sample_index 应在本次 session 严格递增，且在 signed int32 范围内。
with ObservationCollector(env, config) as collector:
    for sample_index in range(1, control_steps + 1):
        env.step(next_action)
        collector.submit(sample_index)       # 捕获当前本地物理状态
        env.render(simulate_index=sample_index, request_idr=True)
        ready = collector.poll()             # 返回完整配对项，可接入下一阶段 adapter
        # 继续循环时须持续 poll；控制器管理超时、停止和自身动作逻辑。
```

示例强调调用顺序，不是已实现的机器人控制器。相机握手是异步的，最初的几个样本可能缺帧；自行 warm-up 后再开始记专家示范。`poll()` 同步写磁盘，当前没有异步 writer；高频控制应在适当采样点采集，测量写盘延迟后再决定是否增加异步存储。外部控制器停止前，可继续 poll 排空网络结果；close 本身不等待或渲染补帧。

SDK 模型桥接在 `StateReader.from_env` 使用 `_mjModel/_mjData`，锁定 26.8.2；更新 SDK 后要重新联调。不要用独立 gRPC 状态诊断代替本地状态：服务端反馈可能滞后，且该响应不带仿真时间/帧号。

## 验收顺序

1. `python -m pip check`，检查当前 SDK/Python 版本。
2. `g1-camera-preview --headless --duration 10 --save-on-exit`：两路分辨率符合布局，帧号递增，原图颜色正确。
3. `g1-robot-state --duration 3 --hz 20`：45 维位置及速度，状态名/单位与当前模型一致。
4. 5 秒统一观测试采集；退出非零或没有样本时先修取流/同步。
5. `g1-dataset <run>`：全部图片能解码、状态 schema/维数正确、图像/状态帧号一致、索引完整。
6. 检查 summary 的 dropped_incomplete、unfinished、camera_queue_overflow。短固定姿态试采集应尽量为零；长时/高分辨率采集需报告实测丢帧率。
7. 接入运动控制器后重新测试实际变化状态和不同姿态图像，不以静止状态采集通过代替运动同步通过。

## 故障处理

- 50051 连接失败：确认 Runtime 已启动并选手动仿真方式。
- 连到相机但不产帧：需要控制器调用 render；仅 receive-only 不产帧。
- Unsupported framing / 缺少 PTS：核对 26.8.2 的 12 字节消息头和完整 access unit；不要降级到最近帧近似配对。
- 相机名有 UUID：默认按短名/前缀唯一匹配；存在多个匹配时用 `--head-name` / `--wrist-r-name` 指定注册名。
- 丢帧/队列溢出：降低采样 fps 或分辨率，检查磁盘吞吐；按实际内存预算调 camera_buffer/pair_capacity。加大容量会增加内存，不能消除持续过载。
- 图片尺寸发生变化：停止采集并启动新 run，避免错误的相机 manifest。
- index 不一致：停采集后运行 `--rebuild-index`；保留原始 run 作为故障证据。
- Windows 中文路径：读写使用 Python 文件接口和内存 PNG 编解码。
