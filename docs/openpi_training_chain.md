# G1 数据接入 OpenPI：本机验证结果

2026-10-05：已将真实采集数据接入 OpenPI 官方 Pi05 调试网络，完成前向损失、反向梯度、3 次优化器更新、检查点保存恢复、动作采样及反归一化。15 项链路检查、4 项数据契约回归测试通过。

**这是随机初始化的调试网络验证，不是预训练 π₀.₅ 微调，不表示已经具备自主按压能力。** 未将预测动作发送到机器人或仿真。

## 数据与实际计算

- 输入：`data/official_demo_24fps/dataset_verified`，1 条成功提交的示范，275 帧，24 FPS。
- 动作窗口：24 个动作（1 秒），共 252 个完整窗口，窗口不跨 episode；不重复对 action 做下一帧偏移。
- 双路图像：头部映射 `base_0_rgb`，右腕映射 `right_wrist_0_rgb`；缺失左腕填零并 mask=false。原始视频保持 1280×960，模型输入采用等比例缩放、黑边填充到 224×224。
- 状态和动作各 18 维：双臂末端基座坐标系位置、xyzw 四元数、夹爪归一化控制值。不是关节角观测。模型内补零到 32 维，输出取前 18 维。
- 使用官方 PaliGemma tokenizer，中文任务指令和归一化状态共同编码为 Pi05 token；最大长度 200。
- 使用官方 Normalize/Unnormalize；q01/q99 尺度下限用于保护单条数据中的近常量维度。统计只来自这条训练示范。尚无独立验证集。
- 在固定的真实第 100 帧及其未来动作窗口上执行 3 步更新；其余起始和末尾窗口通过输入检查。固定噪声、关闭随机图像增强，用于可重复诊断，不代表泛化评估。

| 项目 | 实测 |
|---|---|
| OpenPI commit | `981483dca0fd9acba698fea00aa6e52d56a66c58` |
| 运行设备 | Windows JAX CPU |
| Gemma / action expert | 官方 `dummy` 配置，随机初始化 |
| 视觉编码器 | 官方完整 SigLIP，随机初始化并冻结 |
| 更新参数 | action projections / time MLP，共 12,512 个 |
| 冻结参数 | 429,430,832 个 |
| 第一次更新前 loss | 1.966396 |
| 3 次更新后 loss | 1.773934 |
| 保存恢复后 loss | 1.773934 |
| 梯度范数 | 1.342471、1.311579、1.280799 |
| 采样输出 | `[1, 24, 18]`，有限值 |
| 运行耗时 | 约 98 秒 |

损失采用官方 flow-matching loss。有限预测值仅证明推理计算和数据转换可执行，不证明动作有效、姿态可直接用于控制或任务成功。

## 文件

- `scripts/g1_openpi_data.py`：本地 LeRobot v2.1 视频/Parquet 读取、窗口构造、G1Inputs/G1Outputs、统计。
- `scripts/check_openpi_training.py`：本次实际执行的 CPU 调试训练链路。
- `scripts/train_g1_openpi.py`：完整预训练 Pi05 LoRA 配置及官方 train.py 启动入口；本机仅通过配置构造检查，尚未运行预训练微调。
- `tests/test_g1_openpi_data.py`：右腕映射、无标签推理、动作窗口无二次偏移和 episode 边界、错误维度和非有限输出检查。
- `data/openpi_chain_03/report.json`：最终通过的报告。
- `data/openpi_chain_03/assets/g1_buttons/norm_stats.json`：官方格式统计。
- `data/openpi_chain_03/debug_train_state.msgpack`：仅包含调试网络可训练参数和优化器数组；冻结随机骨干需按 seed=0 和同一配置重建，不能用于正式策略服务。
- `data/openpi_chain_03/debug_predicted_actions.npy`：离线调试预测，禁止当作已训练控制策略。
- `data/openpi_chain_03/*_rgb.png`：真实送入模型的图像，便于检查按钮是否足够清晰。
- `data/openpi_chain_03.log`：训练日志。

`openpi_chain_01` 是依赖兼容性失败尝试；`openpi_chain_02` 在优化器状态序列化处失败，初步报告不代表训练完成。最终只以 `openpi_chain_03/report.json` 中 `training_chain_complete=true` 为准。

## 复现本次 CPU 检查

使用隔离的系统包继承环境，未在 OrcaLab 环境内安装训练依赖。该环境是 Windows CPU 检查环境，不等同于官方完整 Linux uv.lock 环境。

```powershell
# 从本项目目录运行；输出目录必须不存在。
python `
  scripts/check_openpi_training.py `
  --openpi-root ../openpi `
  --dataset data/official_demo_24fps/dataset_verified `
  --output data/openpi_chain_new --steps 3

# 只验证预训练配置，不下载大模型或启动微调。
python `
  scripts/train_g1_openpi.py `
  --openpi-root ../openpi `
  --dataset data/official_demo_24fps/dataset_verified `
  --assets data/openpi_chain_03/assets `
  --checkpoints data/openpi_pretrained_checkpoints
```

重要兼容版本：JAX/jaxlib 0.5.3、Flax 0.10.2、Optax 0.2.4、NumPy 1.26.4、Transformers 4.53.2、Pydantic 2.10.6、numpydantic 1.6.7。最新 Pydantic 与这个 numpydantic 组合发生过 schema 错误，已在隔离环境固定兼容版本。其他实际包版本保存在最终输出的环境清单中。

## 下一阶段：Linux GPU 预训练微调

当前本机 RTX 5080 约 16 GB 显存，未安装可用 WSL。正式微调需要准备独立 Linux GPU 环境与预训练权重。按照比赛文档锁定 OpenPI 981483d，安装官方 uv 环境，并把本项目脚本、数据和统计复制过去。

在 Linux OpenPI 环境运行 `train_g1_openpi.py`，使用对应的绝对路径，添加 `--train`；该入口加载 `gs://openpi-assets/checkpoints/pi05_base/params`，使用双分支 LoRA、32 维模型动作空间、24 步窗口和官方训练主循环。缺省 batch=1、1000 步，仅为首次小数据试验起点。**该正式训练分支尚未实测；权重加载、显存、完整 checkpoint 与策略服务仍待 Linux 验证。**

数据读取采用本地 G1Dataset，绕过 Hub 查找和不同视频后端的差异，仍使用官方数据变换、归一化和训练主循环。当前将压缩至 224×224 的所有视频帧缓存在内存，适合首轮小数据；大规模采集后应改为有界缓存或离线预处理。新数据/训练集划分后必须重新计算统计，不使用测试集计算统计。

后续任务成功评价、相机与物理步时间对齐、控制端四元数归一化及动作范围检查均需另外验证。本次固定样本损失下降不替代这些检查。
