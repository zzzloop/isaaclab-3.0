# AM-DP123 PICO XR 遥操作包（Isaac Lab 外部项目）

本目录是 `amgg_robot_lab` 外部项目的中文文档。项目把 AM-DP123 移动双臂机器人接入
Isaac Lab 官方 `isaaclab_teleop` 遥操作管线，用于 PICO 手柄 + CloudXR 的真机对齐与
遥操作验证，并额外提供一个不依赖真机、PICO 或 Pink IK 的 PI0.5 策略仿真验证任务
（见下文“PI0.5 仿真验证平台”）。

## 目标与路线（路线 A）

* **不改资产**：直接使用未修改的 `AM-DP123.urdf` 与随包发布的 61 个 STL 网格
  （URDF 的 SHA256 被固定在一个离线测试里，任何资产改动都会让测试失败）。
* **不改官方管线**：遥操作入口仍然是官方脚本
  `scripts/environments/teleoperation/teleop_se3_agent.py`，本扩展只通过
  `--external_callback` 注册自己的任务和 IK 动作项。
* **契约先行**：关节顺序、坐标系名、相机名称在 `amgg_robot_lab.contracts` 中集中定义，
  仿真、XR 面板、数据录制和真机共用同一套 ABI。

## 目录结构

```
amgg_robot_lab/
├── .gitattributes                  # *.STL 走 Git LFS
├── pyproject.toml                  # 包构建、依赖与离线检查配置
├── scripts/
│   ├── amgg_teleop.py              # 注入 --external_callback 后调用官方遥操作脚本
│   └── am_dp123_pi05_eval.py       # PI0.5 mock / remote 策略运行与记录入口
├── source/amgg_robot_lab/
│   ├── changelog.d/                # 变更片段
│   ├── README.md / README_CN.md    # 英文 / 中文文档
│   └── amgg_robot_lab/
│       ├── assets/                 # AM-DP123 URDF + STL + 机器人配置
│       ├── contracts/              # 关节 / 坐标系 / 相机契约
│       ├── kinematics/             # 离线 URDF FK / 雅可比 / DLS-IK
│       ├── policy/                 # OpenPI 观测协议 + 动作布局 JSON + 安全适配器
│       ├── tasks/                  # Gym 注册 + 遥操/PI0.5 环境配置 + 自定义 Pink 动作项
│       └── teleop/                 # IsaacTeleop PICO 管线（18 维动作）
└── tests/                          # 10 个离线测试文件
```

## 安装

在 Linux 验证机上，于 Isaac Lab 仓库根目录执行：

```bash
./isaaclab.sh -i                 # 基础环境
./isaaclab.sh -i teleop          # 追加 isaaclab_teleop 依赖（pink / pinocchio / isaacteleop）
uv run python -c "import pink, pin, isaacteleop; print('teleop deps ok')"
```

在 Windows（无 Isaac Sim）上只能跑离线测试与静态检查，用于快速回归：

```powershell
cd amgg_robot_lab
uv run --no-project --with pytest --with numpy python -m pytest -q
uv run --no-project --with ruff ruff check .
uv run --no-project --with ruff ruff format --check .
```

## 运行

```bash
# 无头冒烟：只验证资产/场景/动作项能构建
uv run python amgg_robot_lab/scripts/amgg_teleop.py \
    --task Isaac-AM-DP123-Pico-XR-v0 --viz none \
    --cloudxr_env none --no-auto_launch_cloudxr

# PICO + CloudXR 无头运行：必须保留外部相机（XR 图像面板复用这 4 路相机）
uv run python amgg_robot_lab/scripts/amgg_teleop.py \
    --task Isaac-AM-DP123-Pico-XR-v0 --xr --cloudxr_env cloudxrjs --viz none
```

`amgg_teleop.py` 会先把扩展源码目录加入 `sys.path`，再拒绝重复传入 `--external_callback`，因为它要占用该参数注册
`amgg_robot_lab.tasks:register_tasks`。需要指定显卡时使用官方参数 `--device cuda:N`；包装器不会改写
`CUDA_VISIBLE_DEVICES`。默认不要传 `--disable_external_cameras`：
XR 图像面板（`xr_camera_feeds`）依赖场景里的相机渲染。

## 契约（ABI）

### 关节契约 `amgg_robot_lab.contracts.am_dp123_joint_contract`

| 常量 | 维度 / 内容 |
| --- | --- |
| `AM_DP123_STATE_JOINT_NAMES` / `AM_DP123_STATE_DIM` | 23：腰 3 + 左臂 7 + 右臂 7 + 手 4 + 头 2 |
| `AM_DP123_IK_JOINT_NAMES` | 14：仅左右臂（腰、头、手不参与 Pink IK） |
| `AM_DP123_ABSOLUTE_IK_ACTION_DIM` | 18：[左手基座 xyz+xyzw(7), 右手基座 xyz+xyzw(7), 4 个手动作] |
| `AM_DP123_CONTROLLED_JOINT_NAMES` / `AM_DP123_JOINT_POSITION_ACTION_DIM` | 18：`command_enabled=True` 的关节（腰为被动自由度） |
| `AM_DP123_HAND_JOINT_NAMES` | `left_arm_hand_joint1_0/2_0`, `right_arm_hand_joint1_0/2_0` |
| 手开合 | 张开 `0.0 rad`，闭合 `±0.32 rad`；触发 +1 张开、−1 握紧（`GripperRetargeter` 约定） |
| `AM_DP123_LOCOMOTION_JOINT_NAMES` | 8 个底盘转向 / 驱动轮关节（本任务不控） |

夹爪触发到关节的映射由
`am_dp123_hand_closed_fraction()`、`AM_DP123_HAND_ACTION_SIDE_INDEX = (0, 0, 1, 1)` 与
`AM_DP123_HAND_ACTION_TRIGGER_INDEX = (0, 0, 2, 2)` 定义：左右手输入保持独立，
每只手的一个动作标量同时驱动该手的两根手指。

### 坐标系契约 `AM_DP123_FRAMES`

| 语义名 | URDF 链接 |
| --- | --- |
| `base_link` | `base_link` |
| `torso_link` | `waist_link3` |
| `head_link` | `head_link2` |
| `head_camera_link` | `head_camera_link` |
| `left_wrist_link` / `right_wrist_link` | `left_arm_link7` / `right_arm_link7` |
| `left_hand_base_link` / `right_hand_base_link` | `left_arm_hand_link` / `right_arm_hand_link` |
| `left_wrist_camera_link` / `right_wrist_camera_link` | `left_arm_camera_link` / `right_arm_camera_link` |
| `left_tcp_offset_m` / `right_tcp_offset_m` | 腕部到手基座的固定偏移；Pink 通过 URDF 固定关节纳入求解 |

### 相机契约 `AM_DP123_CAMERAS`

四路相机（`ros` 约定，光轴 +Z、图像上方 −Y），基线 0.06 m：

| 名称 | 父链接 | 平移 [m] | 四元数 xyzw |
| --- | --- | --- | --- |
| `head_left` | `head_camera_link` | (0, 0.03, 0.0288) | (0, 0, −0.70711, 0.70711) |
| `head_right` | `head_camera_link` | (0, −0.03, 0.0288) | (0, 0, −0.70711, 0.70711) |
| `left_wrist` | `left_arm_camera_link` | (0.034, 0, 0) | (0.5, 0.5, 0.5, 0.5) |
| `right_wrist` | `right_arm_camera_link` | (−0.0323, 0, 0) | (0.5, −0.5, −0.5, 0.5) |

四路相机共用同一组内参（分辨率 640×480 @ 30 Hz）：

| 内参 | 值 |
| --- | --- |
| 焦距 `focal_length_mm` | 18.0 mm |
| 感光面宽 `horizontal_aperture_mm` | 20.955 mm |
| 对焦距离 `focus_distance_m` | 1.0 m |
| 裁剪面 `clipping_range_m` | (0.05, 10.0) m |
| 水平 / 垂直 FOV（由上式推导） | 60.41° / 47.17° |

这些内参是**仿真默认值**，不是真机标定结果；外参也来自 URDF 支架几何
（镜头平面推导）。做真机对齐时请用手眼标定值替换数值，但不要改动四个稳定名称。

## 运动学

* `amgg_robot_lab.kinematics.get_am_dp123_kinematics()`：解析 URDF 得到的 `base_link`
  到任意链接的 FK。
* `compute_am_dp123_forward_kinematics()`：指定关节角下的位姿（`base_link` 坐标系）。
* `solve_am_dp123_inverse_kinematics()`：阻尼最小二乘 IK（`IkTarget` / `IkResult`）。
* Pink IK 的笛卡尔目标是 URDF 中的真实左右手基座
  （`left_arm_hand_link` / `right_arm_hand_link`）。腕部到手部约 17 cm 的固定安装偏移和
  ±90° 安装旋转由 URDF/Pinocchio 直接参与求解，不再用手工 TCP 偏移补偿。

home 姿态下的手基座目标由 URDF FK 生成：左侧位于 `base_link` 下
(0.48874, 0.11600, 0.79338) m，右侧位于 (0.45082, −0.09157, 0.77453) m；世界坐标 Z
再加机器人底座生成高度 `AM_DP123_BASE_SPAWN_HEIGHT_M = 0.0729` m。

## 遥操作管线

`build_am_dp123_pico_pipeline()` 返回一个 `OutputCombiner`：

1. Play 后第一帧有效手柄位姿只用于捕获左右控制器原点，输出保持 URDF home，不发生跳变。
2. 后续输出为 `hand_home + controller_delta`；平移 1:1，旋转用
   `q_controller * inverse(q_origin) * q_home` 映射，全部采用 XYZW。
3. Stop 会冻结当前手部目标并重新准备捕获原点；移动手柄后再 Play 可从原位置继续。
4. 手柄跟踪丢失、位姿无效或四元数退化时保持上一目标，避免无效值进入 Pink。
5. 左右手柄模拟扳机直接输出连续夹爪命令，不创建 PICO 不支持的 OpenXR HandTracker。
6. `TensorReorderer` 按 `AM_DP123_ACTION_LAYOUT` 输出 18 维动作。
7. 动作进入 `AmDp123PinkInverseKinematicsAction`：先把手部触发映射成手指关节目标，
   再交给官方 Pink 求解器；`_raw_actions` 保留 PICO 原始 ABI 以便诊断与录制。

`AM_DP123_IDLE_ACTION` 是 18 维参考动作（手基座在 home 位、双手张开），用作无输入时的
保持指令，也可用于回放脚本。

## 环境配置

`AmDp123PicoXrEnvCfg`（任务 `Isaac-AM-DP123-Pico-XR-v0`）要点：

* 场景：AM-DP123 机器人 + 0.78 m 高工作台 + 5 cm 方块 + 放置标记 + 地面 + 两盏灯，
  以及上述 4 路相机（相机父链接由契约给出，因此相机跟随对应的真实连杆）。
* 动作：`mdp.AmDp123PinkInverseKinematicsActionCfg`，`pink_controlled_joint_names`
  为 14 个臂关节，两个 `FrameTask` 分别控制真实左右手基座，开启重力补偿；
  `fail_on_joint_limit_violation=False`（腰部限位较窄，改回 `True` 前需在服务器上重新标定）。
* 求解器参数沿用一代腕位姿 IK 的代价比例：
  `FrameTaskCfg(position_cost=50.0, orientation_cost=1.0, lm_damping=0.1, gain=1.0)`、
  `DampingTaskCfg(cost=0.1)`、`NullSpacePostureTaskCfg(cost=0.02)`。较弱的姿态先验只用于
  消除 7 自由度手臂的冗余，不再把肩肘强行拉回 home。
* 观测：`actions`、`robot_joint_pos/vel`（23 维状态顺序）、左右腕位姿、方块位置，
  以及 4 路 `image_*`（`normalize=False, clone=False`）。
* 仿真：`dt = 1/120 s`、`decimation = 4`、`render_interval = 2`、`device = "cuda:0"`、
  单环境、`env_spacing = 2.5`。
* XR：`xr_camera_feeds` 使用 `head_left`/`head_right` 组成有左右标签的 PiP 面板
  （`mode="horizontal"`, `placement="head_locked"`）。这两路图像都来自 Isaac Sim 相机。

> 当前 Isaac Lab 3.0 的 `XrCameraFeedCfg` 是 XR 图像面板接口：左右相机并排显示，
> 并不是把左图只送左眼、右图只送右眼。若验收标准要求逐眼立体投送，需要另建
> `isaacteleop.viz`（Televiz）会话并共享 OpenXR handles；不能把现有 PiP 描述成真立体。

## PI0.5 仿真验证平台

独立于 PICO 遥操的 PI0.5 策略仿真任务，用于在真机部署前让 PI0.5 策略通过 WebSocket
接入 IsaacLab，在 AM-DP123 URDF、四路相机和物理环境上执行、观察、限位、记录和回放。
它与遥操任务是两个独立入口，不依赖真机、PICO、Pink IK 或 CloudXR。

* 任务：`Isaac-AM-DP123-Pi05-Eval-v0`（`AmDp123Pi05EvalEnvCfg`）。
* 状态 ABI 仍是 23 维（`AM_DP123_STATE_JOINT_NAMES`），执行 ABI 仍是 18 维
  （`AM_DP123_CONTROLLED_JOINT_NAMES`），动作项为官方 `mdp.JointPositionActionCfg`
  （`preserve_order=True`、`use_default_offset=False`、逐关节 `clip`）。
* 频率 30 Hz（`sim.dt = 1/120`、`decimation = 4`、`render_interval = 2`），单环境。
* 模型动作维度由外部 JSON 决定，**不写死 32 维**：
  * `policy/layouts/am_dp123_joint_position_18.json`：已确认的恒等 18 维布局，mock 与通路测试默认使用。
  * `policy/layouts/am_dp123_pi05_32_template.json`：未确认模板，`source_indices` 为 `null`，不能作为
    运行布局。PI0.5 的 32 维含义确认后，只需填写该 JSON 并同步 OpenPI 数据 transform。

### 观测协议

`amgg_robot_lab.policy.build_pi05_observation()` 把 Isaac Lab 观测转换为 OpenPI payload：

| payload key | 内容 |
| --- | --- |
| `observation/image` | `head_left` 640×480 RGB |
| `observation/image_right` | `head_right` RGB（稳定扩展键，双臂/双目） |
| `observation/wrist_image` | `left_wrist` RGB |
| `observation/wrist_image_right` | `right_wrist` RGB（稳定扩展键） |
| `observation/state` | 23 维 `float32` 关节位置，顺序固定 |
| `observation/joint_velocity` | 23 维 `float32` 关节速度（稳定扩展键） |
| `prompt` | 任务指令原文 |

图像统一转为连续 `uint8` RGB（RGBA 去 alpha，浮点 `[0,1]` 映射到 `0–255`）；状态不在客户端
归一化，归一化由 OpenPI 服务端训练配置处理。`--policy remote` 时使用
`openpi_client.image_tools.resize_with_pad(..., 224, 224)` 与 `convert_to_uint8()`；
`openpi_client` 只在 remote 模式延迟导入，不是 `amgg_robot_lab` 的强制依赖。

### 动作安全适配器

`Pi05ActionAdapter` 按布局把 `(T, model_action_dim)` 或 `(model_action_dim,)` 映射为 18 维绝对关节目标：

1. 校验 chunk 非空、无 NaN/Inf、宽度等于 `model_action_dim`；
2. 用 `source_indices` 选出 18 维，应用 `scale`、`offset`（标量或 18 维数组均可）；
3. `absolute_joint_position` 直接取目标，`delta_joint_position` 从上一目标累加；
4. 按 `AM_DP123_JOINT_SPECS` 做位置限位；
5. 按 `max_velocity_rad_s * control_dt * velocity_scale` 做逐步速度限位；
6. reset 后从当前关节位置起限速，不从零开始；协议或推理失败时保持当前目标。

### 运行命令

mock_hold（无头，验证环境 / 四路相机 / 观测 / 适配 / 步进 / 记录）：

```bash
conda activate isaaclab30
cd ~/zzk_data/IsaacLab

./isaaclab.sh -p amgg_robot_lab/scripts/am_dp123_pi05_eval.py \
    --policy mock_hold --max_steps 120 --viz none --device cuda:0 \
    --kit_args "--/renderer/multiGpu/enabled=false"
```

mock_sine（肉眼确认策略动作确实驱动机器人）：

```bash
./isaaclab.sh -p amgg_robot_lab/scripts/am_dp123_pi05_eval.py \
    --policy mock_sine --max_steps 600 --viz kit --device cuda:0 \
    --kit_args "--/renderer/multiGpu/enabled=false"
```

remote（OpenPI 服务端与 IsaacLab 客户端使用独立环境 / GPU）：

```bash
# 服务端（独立环境）
cd ~/openpi
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=<AM_DP123_CONFIG> --policy.dir=<CHECKPOINT_PATH>

# IsaacLab 客户端
conda activate isaaclab30
cd ~/zzk_data/IsaacLab
./isaaclab.sh -p amgg_robot_lab/scripts/am_dp123_pi05_eval.py \
    --policy remote --host 127.0.0.1 --port 8000 \
    --prompt "pick up the orange cube and place it on the green target" \
    --action_layout amgg_robot_lab/source/amgg_robot_lab/amgg_robot_lab/policy/layouts/<CONFIRMED_LAYOUT>.json \
    --action_horizon 8 --viz none --device cuda:0 \
    --kit_args "--/renderer/multiGpu/enabled=false"
```

> 32 维含义确认前，remote 模式只完成网络、shape 与安全失败验证，不能据此宣称能正确控制
> 机器人。推理连续失败 `MAX_CONSECUTIVE_INFERENCE_FAILURES` 次后安全退出，不会无限高速重试。

### 记录格式

默认目录 `outputs/am_dp123_pi05_eval/<timestamp>/`，每个 episode 保存：

* `episode_XXXXXX.npz`：`joint_pos (N,23)`、`joint_vel (N,23)`、`object_position (N,3)`、
  `model_action (N,model_action_dim)`、`applied_joint_target (N,18)`、`terminated (N,)`、`truncated (N,)`。
* `episode_XXXXXX.json`：task id、prompt、policy mode、remote host/port、action layout 路径与
  SHA-256、state/controlled 关节名、control dt、Git commit、步数与终止原因。
* `--record_images` 时另存 `episode_XXXXXX_images.npz`：默认每 15 步一帧（`--image_stride`），
  最多 120 帧，避免无界内存积累；不引入视频编码依赖。

### 真机部署前必须统一

动作布局（`source_indices` 与模式）、归一化统计（state/action 均值方差）、相机标定（内参与手眼
外参）、控制频率与延迟。当前 18 维布局与四路相机参数是**仿真配置，不是真机标定结果**；
也不得把当前仿真相机参数描述为真机标定结果。

## 离线测试与静态检查

`tests/` 下的 10 个文件只依赖 `numpy` + 标准库（在 Windows 上即可运行），覆盖：

| 测试文件 | 覆盖内容 |
| --- | --- |
| `test_am_dp123_urdf_contract.py` | URDF 结构、链接/关节名、URDF SHA256 固定值 |
| `test_am_dp123_joint_contract.py` | 关节 ABI 维度、限位、home 位、手部映射 |
| `test_am_dp123_camera_contract.py` | 相机外参落在支架镜头平面、基线、内参与 FOV |
| `test_am_dp123_hand_grasp.py` | 触发量到手指关节的开合映射 |
| `test_am_dp123_kinematics.py` | 离线 FK / 雅可比 / DLS-IK 精度与限位裕度 |
| `test_am_dp123_pico_retargeting.py` | Play 零点捕获、相对姿态映射、掉帧保持、双手基座 IK 可达性 |
| `test_am_dp123_action_abi.py` | 18 维动作布局与 PICO 管线的元素顺序 |
| `test_am_dp123_development_import.py` | 包可导入、发布资产存在 |
| `test_am_dp123_pi05_protocol.py` | OpenPI 观测 payload、动作布局校验、位置/速度限位与 delta/absolute 模式 |
| `test_am_dp123_pi05_registration.py` | PI0.5 任务注册、直接关节位置动作项、与 Pink/IsaacTeleop 解耦、布局打包 |

```bash
cd amgg_robot_lab
uv run --no-project --with pytest --with numpy python -m pytest -q
uv run --no-project --with ruff ruff check . && uv run --no-project --with ruff ruff format --check .
```

## 服务器校验清单

1. `./isaaclab.sh -i teleop`，并确认 `import pink, pinocchio, isaacteleop` 成功。
2. 无头冒烟：`uv run python amgg_robot_lab/scripts/amgg_teleop.py --task Isaac-AM-DP123-Pico-XR-v0 --viz none --cloudxr_env none --no-auto_launch_cloudxr`
   （确认资产导入、Pink 控制器构建、4 路相机创建成功）。
3. XR：加 `--xr --cloudxr_env cloudxrjs --viz none`，**不要**加 `--disable_external_cameras`。
   正常遥操不要加 `--enable_debug_visualization`。运行配置不再创建 retargeter 调参面板，PICO
   视野中不会出现该面板的空白文字框和两条三角形交互射线。
4. 戴好头显后先把两只手柄放在舒适中立位，再点 Play。第一帧只建立原点，机器人应保持 home；
   随后分别前后、左右、上下移动 3–5 cm，确认同侧手基座同方向运动，再验证旋转与夹爪。
   需要重新摆放手柄时先 Stop，摆好后再 Play，机器人应从停止位置连续恢复。
5. PI0.5 冒烟（独立入口）：`./isaaclab.sh -p amgg_robot_lab/scripts/am_dp123_pi05_eval.py --policy mock_hold --max_steps 120 --viz none --device cuda:0 --kit_args "--/renderer/multiGpu/enabled=false"`，
   确认四路相机创建、机器人保持 home、120 step 后正常退出并生成 NPZ/JSON（无 NaN、shape 正确）；
   再用 `--policy mock_sine --max_steps 600 --viz kit` 确认机器人平滑小幅运动、无越限或仿真发散。
6. 提交前运行 `uv run isaaclab -f`（ruff + pre-commit 全量检查）。

## 已知限制

* 相机内参是仿真默认值；外参来自 URDF 支架几何，不是手眼标定结果。
* `left/right_tcp_offset_m` 仍供硬件和诊断代码读取；仿真 Pink 直接控制 URDF 手基座帧。
* 腰部在 Pink IK 中不被控制（`command_enabled=False`）；任务假设腰部保持 home 位。
* 夹爪触发是连续量映射到 ±0.32 rad，尚未做力控/滑移建模。
* `XrCameraFeedCfg` 显示的是双路 PiP；逐眼立体图像需要 Televiz 专用集成。

## 版本与迁移

* 变更片段位于 `changelog.d/codex-am-dp123-pico-xr-impl.minor.rst`。
* 命名迁移：本扩展使用 AM-DP123 的关节、坐标系、相机名称（`amgg_robot_lab.contracts.am_dp123_*`），
  下游代码应改用这些契约，而不是旧的 AMGG 命名。
* 61 个 `*.STL` 通过 `amgg_robot_lab/.gitattributes` 走 Git LFS；把大网格提交进仓库前请确认
  `git lfs` 已安装（`check_git_lfs_pointers` 预提交钩子会校验索引里存的是指针而不是原始二进制）。
