# 统一目标、解析与成功判定（第一版）

本版完成可执行的目标协议、受控文字/框选解析、流式开发判定器，并对现有真实采集与回放物理记录做了离线验证。20 项测试通过。没有启动机器人运动、修改原始示范标签或宣称官方裁判成功。

## 使用边界

- 文字解析使用配置中明确的名称/别名，支持单条正向操作指令；不是通用语言模型。模糊描述、否定/多步骤指令、重复目标、错误操作会报错。
- 框选解析依赖带来源的候选检测框。本版 `task_catalog.json` 中的框来自用户参考图片的人工核对，**尚未实现自动目标检测/OCR**。更换图片须由检测器/OCR或人工标注提供新候选框与 image_id，不能复用旧坐标。
- `image_id + camera` 必须完全匹配；归一化 xyxy 框须有效，并唯一覆盖一个候选元件。框内可包含铭牌，但候选框是按钮本体。框和名字仅确定对象，不产生基座坐标系三维接触位姿，`contact_pose_b=null`。
- `cabinet02/03` 是当前开发实例标识，不是已核对的官方赛题编号。仿真 joint 绑定只用于判定，不进入策略输入。
- 当前按压开发标准：相对首帧沿正向位移 ≥1.5 mm，持续 ≥80 ms，然后回到初值 ±0.5 mm。**数值和位移方向尚未物理标定，不是官方成功阈值。** 设置位于 `task_success_dev.json`。
- 旋钮/拨杆要求显式 `angle_deg`（15～180）及已标定的 `joint_direction`（±1）。请求角度进入目标身份及判定阈值，不会将 90° 请求按 15° 判定。顺/逆时针到关节方向的映射尚未实现，相关自然语言被拒绝。角度事件只表示达到相对初始位置的指定变化，不保证保持最终状态。

## 配置和代码

| 文件 | 用途 |
|---|---|
| `configs/task_catalog.json` | 对象身份、名称、类型、参考图检测和仿真绑定 |
| `configs/goal_stop_text.json` | “按压停止按钮”的文字请求 |
| `configs/goal_stop_roi.json` | 用户参考图的框选请求及 press 操作 |
| `configs/task_success_dev.json` | 带来源的开发判据和非目标状态监测阈值 |
| `scripts/task_goal.py` | 解析、唯一目标身份、OpenPI 语言输入桥接 |
| `scripts/task_success.py` | 每集重置的流式判定状态机 |
| `scripts/run_task_goal.py` | 离线解析/判定 CLI，不发送控制命令 |
| `tests/test_task_goal.py` | 20 项契约和反例测试 |

统一目标包含 schema_version、goal_id、target_id、cabinet_id、target_name、target_type、operation、desired_result、policy_prompt、grounding 及定位状态。相同对象与操作的文字/ROI 输入得到同一个 goal_id。grounding 保留文字/图像来源用于审计。

## 运行

从项目根目录运行；输出文件使用独占创建，不覆盖已有结果：

```powershell
python `
  scripts/run_task_goal.py `
  --catalog configs/task_catalog.json `
  --request configs/goal_stop_text.json `
  --criteria configs/task_success_dev.json `
  --trace data/official_demo_24fps/capture_verified_audit/physics_trace.jsonl `
  --output data/task_goals_new/capture_text.json
```

将 request 换成 `configs/goal_stop_roi.json` 即使用参考框选。只解析目标时同时省略 `--criteria` 和 `--trace`。

## 已有数据验证结果

- 最终结果：`data/task_goals_v1/capture_text.json` 与 `data/task_goals_v1/replay_roi.json`。
- 两种输入解析到 `cabinet02.stop`，同一个 goal_id。
- 两次均为 `proxy_pass`，`official_success=null`。
- 采集记录含 276 条物理采样：约 6.755 秒确认持续位移，7.755 秒观察到回弹。此数量与 275 对相邻状态/动作标签不矛盾。
- 非目标状态没有超过当前配置的监测阈值；这不等于完全没有漂移或碰撞。

状态包括 `in_progress`、结束后 `incomplete`、`proxy_pass`、`non_target_change`、`timeout`、`inconclusive`。数据缺失、非有限关节数值、超过 150 ms 的采样缺口会使该集不可判定；重复/乱序时间戳报错。短时尖峰不会触发持续按压；按压事件锁存一次，回弹不会丢失成功事件，重复按压不会重复发成功奖励。本模块不计算强化学习奖励。

首帧是基线，必须在动作开始前采样；每集新建 evaluator 或调用 reset。按钮已经被按住时开始记录、目标配置错配等情况不能可靠恢复真实基线。检测到非目标位移只报告状态变化，不凭总 contact_count 推断碰撞责任。撤回、摔倒、抖动和官方判定尚未接入，报告保留为未知。

## 接入后续采集与 OpenPI

`run_official_demo.py` 增加三个可选参数（需要一起提供）：

```text
--goal-request configs/goal_stop_text.json
--goal-catalog configs/task_catalog.json
--goal-criteria configs/task_success_dev.json
```

这些参数放在转发给官方脚本的 `--` 之前。采集时以相同物理审计采样喂给判定器，结束后在 audit-output 中写入 `task_evaluation.json`。本轮使用离线真实轨迹验证状态机，并对接入代码做编译检查，尚未重新执行含该参数的现场采集。

策略输入桥接示例：

```python
from task_goal import resolve_goal, policy_observation
goal = resolve_goal(request, catalog)
obs = policy_observation(camera_and_robot_observation, goal)
# obs['prompt'] 可交给 G1Inputs；不添加场景 joint 真值。
```

新示范中应保存 goal 作为 episode 旁路元数据，并在训练和推理端使用同一 policy_prompt。当前桥接生成带 cabinet_id 的指令，不能只在旧模型推理时改变语言格式并假定效果不变；旧数据原任务文本没有被修改。

下一项感知工作是：检测按钮本体与铭牌、OCR/视觉语义匹配、跟踪当前视角中的目标，并通过深度或标定得到可用接触位姿。当前受控解析接口已经为新检测结果留好输入位置，但不把人工参考框当作自动识别能力。
