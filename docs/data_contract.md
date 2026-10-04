# 数据契约 v1

目录格式保持原采集器的 `schema_version: 1`。本次新增 manifest 配置/依赖信息与 summary 退出状态，原字段保持可读取。

## manifest.json

`state_manifest` 声明 schema_id、实际模型关节名、固定顺序、分组、关节地址、限位与单位。`schema_id` 来自原状态 manifest 的哈希，包含模型地址和来源；不同模型布局地址即使语义相同也可能产生不同 ID，模型 adapter 应检查名字和单位，不仅检查向量长度。

`cameras` 声明实际注册相机名、宽高与端口。采集器保留 UUID 后缀，不假设仅有固定短名。`resolved_config` 是启动时实际配置，`provenance` 记录工程版本、Python、系统及关键依赖版本。`physics_stepped_by_collector: false`、`has_actions: false` 明确当前能力。

## observation.json

- `sample_id`：运行内保存序号与仿真帧号，序号可能不等于每帧的 sequence。
- `prompt`：采样时目标指令，可为每个 submit 单独指定。
- `observation.state`：45 / 30 / 15 维原始实测位置；`joint_velocity` 同序存储。当前 G1 为 rad / rad/s。未来滑动关节必须保持 manifest 中的 m / m/s。
- `simulation_time_s`：本地 MuJoCo 时间。固定姿态采集不会推进它，所以可保持不变。
- `simulate_index`：调用者控制的非负 signed int32 标签，运行内必须严格递增；用于 render、图像帧头和状态配对。不是必然等于物理积分步数。
- `sequence`：采样提交序号；缺帧和抽样会导致保存序号与它不连续。
- `images`：每路图像的相同 simulate_index、原始编码时间戳、主机接收时间、实际尺寸和相对 PNG 路径。
- `source_timestamp_raw`：服务端原始 uint64，单位/时钟未在当前接口中确定，不当成秒或跨主机时间。
- `received_monotonic`：主机单调时钟，用于进程内超时，不用于跨机器对时。
- `action` / `annotation`：当前为 null，保留未来扩展位置，但没有已实现的动作写入 API。

## 同步保证与边界

严格采集只接受 **12 字节小端帧头**（uint64 timestamp + int32 index）及每条消息一个完整 H.264 access unit。使用包 PTS 跟踪解码后的图像，即使解码缓冲或 B 帧重排，也不使用最后收到的消息头标记旧画面。无法追踪 PTS 时拒绝近似对齐并重连。

相机诊断预览可识别旧 8 字节/裸 Annex-B 格式，其元数据是近似的，不用于训练。两路窗口看起来同步或“同时 snapshot”不等于严格同步。

`PairBuffer` 按提交顺序输出。相机先到下一帧时等待前帧，前帧超时后丢弃。相机队列溢出与不完整配对计数分开；CLI 无完整观测超过 timeout 时返回非零。调用者必须在同一控制线程按“完成物理步 → submit → 对应 render → poll”的顺序运行；不得在两个控制器里重置环境。

## 写入与恢复

每个样本先写入 `.partial_<id>`，完成 PNG 与 JSON 后在同一文件系统内 rename 为完整目录，之后追加 index.jsonl。文件先 flush/fsync，临时 JSON 用替换提交。正常写入失败会清理本次未提交目录，已提交样本保留。

这是单进程、单 writer 设计；不支持并发 writer、自动续采或从中途恢复同一个 run。异常退出可留下 partial 目录或落后的 index。停止采集后执行 `g1-dataset <run> --rebuild-index`：校验完整样本并备份旧索引，原始数据和 partial 目录不会被删除。索引恢复不能修复已损坏图片。

rename/fsync 的断电行为依赖文件系统，不声明 Windows 网络盘或云同步盘具备强事务保证。建议采集写本地磁盘，结束并校验后再复制/同步。

## 兼容性规则

可选元数据扩展不改变 schema_version；变更状态顺序、单位、同步含义或必需字段时，需要版本迁移和回归测试。模型训练工件应记录采集版本、数据 schema、profile、相机键、归一化统计和对应代码提交。
