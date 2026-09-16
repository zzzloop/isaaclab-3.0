# AM-DP123 PICO XR 遥操作包（Isaac Lab 外部项目）

本目录是 `amgg_robot_lab` 外部项目的中文文档。项目把 AM-DP123 移动双臂机器人接入
Isaac Lab 官方 `isaaclab_teleop` 遥操作管线，用于 PICO 手柄 + CloudXR 的真机对齐与
遥操作验证。

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
│   └── amgg_teleop.py              # 注入 --external_callback 后调用官方遥操作脚本
├── source/amgg_robot_lab/
│   ├── changelog.d/                # 变更片段
│   ├── README.md / README_CN.md    # 英文 / 中文文档
│   └── amgg_robot_lab/
│       ├── assets/                 # AM-DP123 URDF + STL + 机器人配置
│       ├── contracts/              # 关节 / 坐标系 / 相机契约
│       ├── kinematics/             # 离线 URDF FK / 雅可比 / DLS-IK
│       ├── tasks/                  # Gym 注册 + 环境配置 + 自定义 Pink 动作项
│       └── teleop/                 # IsaacTeleop PICO 管线（18 维动作）
└── tests/                          # 7 个离线测试文件
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

## 离线测试与静态检查

`tests/` 下的 8 个文件只依赖 `numpy` + 标准库（在 Windows 上即可运行），覆盖：

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
5. 提交前运行 `uv run isaaclab -f`（ruff + pre-commit 全量检查）。

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
