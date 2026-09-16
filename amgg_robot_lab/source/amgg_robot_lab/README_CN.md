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
| `AM_DP123_ABSOLUTE_IK_ACTION_DIM` | 18：[左腕 xyz+xyzw(7), 右腕 xyz+xyzw(7), 4 个手动作] |
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
| `left_tcp_offset_m` / `right_tcp_offset_m` | 腕部到指尖的 TCP 偏移，仅记录，Pink 目标是腕部 |

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
* Pink IK 的笛卡尔目标只针对左右腕（`left_wrist_link` / `right_wrist_link`），
  目标位姿来自 PICO 手柄，经 `Se3AbsRetargeter` 与
  `AM_DP123_LEFT_WRIST_TARGET_OFFSET_DEG` / `AM_DP123_RIGHT_WRIST_TARGET_OFFSET_DEG`
  转换。

home 姿态：腕部目标位于 `base_link` 下 (0.36, ±0.20, 0.88) m，机器人底座生成高度
`AM_DP123_BASE_SPAWN_HEIGHT_M = 0.0729` m。

## 遥操作管线

`build_am_dp123_pico_pipeline()` 返回 `(OutputCombiner, retargeters)`：

1. `ControllersSource` → `Se3AbsRetargeter`，输出左右腕绝对位姿。
2. 左右手柄模拟扳机直接输出连续夹爪命令，不创建 PICO 不支持的 OpenXR HandTracker。
3. `TensorReorderer` 按 `AM_DP123_ACTION_LAYOUT` 输出 18 维动作。
4. 动作进入 `AmDp123PinkInverseKinematicsAction`：先把手部触发映射成手指关节目标，
   再交给官方 Pink 求解器；`_raw_actions` 保留 PICO 原始 ABI 以便诊断与录制。

`AM_DP123_IDLE_ACTION` 是 18 维参考动作（腕部在 home 位、双手张开），用作无输入时的
保持指令，也可用于回放脚本。

## 环境配置

`AmDp123PicoXrEnvCfg`（任务 `Isaac-AM-DP123-Pico-XR-v0`）要点：

* 场景：AM-DP123 机器人 + 0.78 m 高工作台 + 5 cm 方块 + 放置标记 + 地面 + 两盏灯，
  以及上述 4 路相机（相机父链接由契约给出，因此相机跟随对应的真实连杆）。
* 动作：`mdp.AmDp123PinkInverseKinematicsActionCfg`，`pink_controlled_joint_names`
  为 14 个臂关节，`target_eef_link_names` 为左右腕，开启重力补偿；
  `fail_on_joint_limit_violation=False`（腰部限位较窄，改回 `True` 前需在服务器上重新标定）。
* 求解器参数：`FrameTaskCfg(position_cost=8.0, orientation_cost=1.0, lm_damping=10.0, gain=0.45)`、
  `DampingTaskCfg(cost=0.4)`、`NullSpacePostureTaskCfg(cost=0.35)`。
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

`tests/` 下的 7 个文件只依赖 `numpy` + 标准库（在 Windows 上即可运行），覆盖：

| 测试文件 | 覆盖内容 |
| --- | --- |
| `test_am_dp123_urdf_contract.py` | URDF 结构、链接/关节名、URDF SHA256 固定值 |
| `test_am_dp123_joint_contract.py` | 关节 ABI 维度、限位、home 位、手部映射 |
| `test_am_dp123_camera_contract.py` | 相机外参落在支架镜头平面、基线、内参与 FOV |
| `test_am_dp123_hand_grasp.py` | 触发量到手指关节的开合映射 |
| `test_am_dp123_kinematics.py` | 离线 FK / 雅可比 / DLS-IK 精度与限位裕度 |
| `test_am_dp123_action_abi.py` | 18 维动作布局与 PICO 管线的元素顺序 |
| `test_am_dp123_development_import.py` | 包可导入、发布资产存在 |

```bash
cd amgg_robot_lab
uv run --no-project --with pytest --with numpy python -m pytest -q   # 50 passed
uv run --no-project --with ruff ruff check . && uv run --no-project --with ruff ruff format --check .
```

## 服务器校验清单

1. `./isaaclab.sh -i teleop`，并确认 `import pink, pinocchio, isaacteleop` 成功。
2. 无头冒烟：`uv run python amgg_robot_lab/scripts/amgg_teleop.py --task Isaac-AM-DP123-Pico-XR-v0 --viz none --cloudxr_env none --no-auto_launch_cloudxr`
   （确认资产导入、Pink 控制器构建、4 路相机创建成功）。
3. XR：加 `--xr --cloudxr_env cloudxrjs --viz none`，**不要**加 `--disable_external_cameras`。
   正常遥操不要加 `--enable_debug_visualization`；该参数会在 PICO 画面中加入手柄坐标轴和手部标记。
   先确认机器人正立、左右相机图像方向正确，再分别闭合左右扳机确认两只手互不串扰。
4. 用手柄实测并微调腕部对齐：`AM_DP123_LEFT_WRIST_TARGET_OFFSET_DEG` /
   `AM_DP123_RIGHT_WRIST_TARGET_OFFSET_DEG`（也可通过 IsaacTeleop 的 retargeter 调参 UI 实时调整）。
5. 提交前运行 `uv run isaaclab -f`（ruff + pre-commit 全量检查）。

## 已知限制

* 相机内参是仿真默认值；外参来自 URDF 支架几何，不是手眼标定结果。
* 腕部 TCP 偏移（`left/right_tcp_offset_m`）仅作记录，Pink 目标是腕部关节坐标系。
* 腰部在 Pink IK 中不被控制（`command_enabled=False`）；任务假设腰部保持 home 位。
* 夹爪触发是连续量映射到 ±0.32 rad，尚未做力控/滑移建模。
* `XrCameraFeedCfg` 显示的是双路 PiP；逐眼立体图像需要 Televiz 专用集成。

## 版本与迁移

* 变更片段位于 `changelog.d/codex-am-dp123-pico-xr-impl.minor.rst`。
* 命名迁移：本扩展使用 AM-DP123 的关节、坐标系、相机名称（`amgg_robot_lab.contracts.am_dp123_*`），
  下游代码应改用这些契约，而不是旧的 AMGG 命名。
* 61 个 `*.STL` 通过 `amgg_robot_lab/.gitattributes` 走 Git LFS；把大网格提交进仓库前请确认
  `git lfs` 已安装（`check_git_lfs_pointers` 预提交钩子会校验索引里存的是指针而不是原始二进制）。
