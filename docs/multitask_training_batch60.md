# 首批 60 条示范的多任务训练准备

2026-10-06 状态：**准备完成，正式训练未启动，没有新的训练检查点。**

## 已完成

- 训练集 48 条（每类 16 条）、16,323 帧；独立验证集 12 条（每类 4 条）、3,978 帧。
- 按源数据目录检查划分交叉，检查训练集状态、动作、时间戳及帧号。
- 仅用训练集计算 18 维状态与动作的均值、标准差、q01/q99；统计方法与既有 G1Dataset 相同，近常量维度尺度下限为 0.001。
- 保存并通过官方 OpenPI Normalize 文件格式的读写一致性检查。
- 用新统计通过固定提交 `981483dca0fd9acba698fea00aa6e52d56a66c58` 的 Pi05 双分支 LoRA 配置构造检查：batch=1、1,000 步、24 步动作窗口、模型 32 维/机器人 18 维。
- 两项新增测试通过：划分交叉会拒绝；验证集数值不会用于计算统计。

## 当前阻塞

实际检查：Windows，RTX 5080 16,303 MiB 总显存；WSL 未安装；当前 OpenPI JAX 只返回 CPU 设备。

固定版本 README 的 Hardware Requirements 写明 LoRA 显存估计大于 22.5 GB，测试系统为 Ubuntu 22.04，不支持其他系统。项目正式入口要求 Linux + JAX GPU。因此本次没有启动预训练微调，也没有用随机初始化 dummy 网络替代正式训练。16 GB 的其他低显存实验方案尚未验证，不能据此断言所有模型都无法在本机训练。

继续当前正式 OpenPI 路径，需要可用的 Linux NVIDIA GPU 环境，显存满足该版本要求（通常至少 24 GB，并需实测）。仅安装 WSL 不会增加显存。用户目前仅有这台 Windows 电脑，未创建付费云资源、安装 WSL 或下载大型预训练权重。

## 输出

- `data/multitask_training_20261006/assets/g1_buttons/norm_stats.json`
- `data/multitask_training_20261006/preparation.json`：来源与哈希，明确 `prepared_not_trained`。
- `data/multitask_training_20261006/config_inspection.json`：明确 `inspect_only`，设备为 CPU。
- `scripts/prepare_multitask_training.py`：不解码全部图像即可重新生成训练专用统计。

## 迁移后启动正式训练

将本项目的 scripts、训练统计和 `data/batch_20261005` 连同完整数据迁移，保持数据目录相对关系；清单内部为相对路径。安装固定提交对应的官方 Linux uv 环境。以下从本项目根目录运行，其中 Python 应为该 Linux OpenPI 环境的 Python，`/path/to/openpi` 替换为其实际路径：

```bash
python scripts/train_g1_openpi.py \
  --openpi-root /path/to/openpi \
  --dataset data/batch_20261005/manifest_train.json \
  --assets data/multitask_training_20261006/assets \
  --checkpoints data/openpi_pretrained_checkpoints \
  --experiment g1_multitask_batch60_v1 \
  --steps 1000 --batch-size 1 --train
```

入口会加载 `gs://openpi-assets/checkpoints/pi05_base/params`。目前 Linux 分支尚未实测，权重下载、实际显存、检查点与训练结果仍需验证。现有入口不自动计算独立验证集损失；正式运行时还需实现检查点评估，使用留出的验证清单与训练统计，不能将训练损失当作任务成功率。

若必须保持当前 Windows/16 GB 条件，可另行选择支持本机的小型多任务策略，但这会改变模型方案，需要与预训练 OpenPI 微调结果区分。
