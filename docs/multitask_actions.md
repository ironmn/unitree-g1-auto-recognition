# 按压、旋转与拨杆多任务数据

本轮为现有按压数据补充了旋钮和拨杆示范，统一为 24 FPS、两路 1280×960 视频、18 维绝对末端状态/动作。训练按原始任务文本条件化，未将不同动作混为同一标签。

| 任务 | 数据目录（项目内） | 帧数 | 24 步完整窗口 | 实测目标变化 |
|---|---|---:|---:|---|
| 按压停止按钮 | data/official_demo_24fps/dataset_verified | 275 | 252 | 按压并回弹，开发代理判据通过 |
| 旋转旋钮 | data/multitask/rotate_clearance/dataset_verified | 431 | 408 | 约 -84.3°，15° 代理判据通过 |
| 拨动拨杆式按钮 | data/multitask/toggle/dataset_verified | 311 | 288 | 约 +80.7°，15° 代理判据通过 |

合计 3 集、1017 帧、948 个独立完整窗口。每个训练采样周期按任务轮流取样，短任务循环补齐到每类 408 个窗口，共 1224 个采样位置。补齐没有增加独立数据或场景多样性。每个窗口严格位于单集内，保留相机视图和原始 task 文本。统计由这三条训练示范共同计算，不复用原单任务统计。

## 旋钮撤回修正与质量门槛

原官方旋钮轨迹完成旋转后，在撤回阶段造成急停按钮和行程开关超过开发阈值的状态变化。它保存在 `data/multitask/rotate`，**未纳入最终训练清单**。

保留官方接近、抓取和旋转路点，复制出 `configs/waypoint_rotate_clearance.yaml`：松手后先沿基座 x 方向后退，再横向下降，最后恢复手部姿态。重新采集的 `rotate_clearance` 通过当前开发判据；原官方 YAML 没有修改。这是本场景的撤回适配，不是经过所有布局验证的通用规划器。

`configs/multitask_g1.json` 是最终训练清单。加载时校验每个来源的原始任务文本和对应目标的 `proxy_success`，拒绝开发判定未通过的记录。按钮阈值和非目标阈值仍属于开发代理；三类任务的 `official_success` 均未知，不能当成官方得分结论。

官方现有动作文件名为 press、rotate、toggle，其中 toggle 的原始文本是“拨动拨杆式按钮”。本轮操作的是该文件对应的场景元件，没有另行假设它等同于所有类型的空气开关；其他位置或结构的空气开关需要独立的目标绑定和示范。

## 复现新增采集

从项目根目录运行；输出目录必须是新目录：

```powershell
.\scripts\run_official_demo_24fps.ps1 -Mode Collect -Action rotate `
  -WaypointOverride .\configs\waypoint_rotate_clearance.yaml `
  -RunDirectory .\data\new_rotate
.\scripts\run_official_demo_24fps.ps1 -Mode Collect -Action toggle `
  -RunDirectory .\data\new_toggle
```

`-Action press` 仍可采集原按压示范。旋钮 override 省略时使用原官方路点，存在上文已观察到的撤回问题。

开发判定请求：`configs/goal_rotate_demo.json`（关节负方向 15°）和 `configs/goal_toggle_demo.json`（正方向 15°），方向取自本场景实测。实际采集角度约 84°/81°，不是训练了任意指定角度控制。

## 模型接入与验证范围

`G1MultiTaskDataset` 通过清单聚合独立 LeRobot 数据，不移动或覆盖源数据。`check_openpi_training.py` 和 `train_g1_openpi.py` 均可将 `--dataset` 指向清单。正式训练配置会使用 `pi05_g1_multitask_lora` 名称。

```powershell
python `
  scripts/check_openpi_training.py `
  --dataset configs/multitask_g1.json `
  --openpi-root ../openpi `
  --output data/multitask/new_training_check --steps 3
```

调试网络每步对三类任务各取一个真实窗口，逐个前向/反向计算并平均梯度，再更新参数。首次 batch=3 同时计算触发 CPU 内存不足，失败记录位于 `data/multitask/training_check`；最终使用 batch=1 梯度累积，结果以 `data/multitask/training_verified/report.json` 的完成字段为准。

最终报告、检查点、三条预测动作和重算的统计位于 `data/multitask/training_verified`。实测每步累积三任务梯度，共 3 次优化器更新，平均固定样本损失从 1.992046 降到 1.800602；保存恢复后的损失一致，采样输出为 `[3,24,18]`。15 项链路检查通过，另有 3 项多任务回归测试和 4 项原 G1 数据契约测试通过。整个调试验证约 174 秒。

这些仍是随机初始化的 OpenPI 调试网络，仅更新动作侧小层，**不是完整预训练 π₀.₅ 的微调结果**。正式 Linux GPU 训练入口已接受清单并通过配置检查，但仍需对应硬件与预训练权重验证。

新增示范增加了动作种类，还没有证明泛化提升。每类只有一集、一个布局；下一阶段需要变化目标位置、初始手势、光照等，并用独立测试场景比较三任务成功率、误操作率和语言指令切换能力。
