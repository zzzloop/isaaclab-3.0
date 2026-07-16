# AMGG Robot Lab：仿真、PICO 遥操、自动评测与 LeRobot 数据闭环

本目录是与 Isaac Lab 核心代码隔离的 AMGG 机器人研究工程。当前已经包含：原始模型留档、可重复生成的标准化 URDF、Isaac Lab 本体配置、离线 FK/Jacobian/双臂 IK、PICO 控制器遥操、四个任务、自动成功/失败判定、官方 HDF5 录制入口、LeRobot Dataset v3 转换器、四相机数据契约，以及真机 ROS 2 接口边界。

本机没有完整 Isaac Sim 运行时，因此本机能完成的是模型、数值运动学、数据格式和代码静态验收；最终的 PhysX、Pink、渲染、PICO 和相机方向必须在服务器 `lab6` 环境按本文“服务器验收”章节逐项运行。没有经过服务器验收前，不应把当前参数描述为已完成物理标定。

## 1. 重要边界

- `amgg_robot_raw.urdf` 是收到的 SolidWorks 导出原件，不修改，用于审计和复现。
- `amgg_robot.urdf` 由脚本生成，是仿真、Pink 和离线 FK/IK 的统一模型。
- 原始模型没有夹爪。当前左右平行夹爪是明确标注的“仿真研究末端”，用于先跑通抓取、任务评测和数据链路，不等同于真实硬件。
- 真实夹爪 CAD、质量惯量、TCP、开合范围和控制协议到位后，只替换末端层和对应契约，不改任务与数据集公共 key。
- 真机 ROS 2 后端不会替代急停、驱动器使能、碰撞监控或厂商安全控制器。

## 2. 目录和职责

```text
amgg_robot_lab/
├── scripts/
│   ├── amgg_prepare_robot_asset.py       # 从原始 ROS 包生成标准化资产
│   ├── amgg_check_robot_asset.py         # URDF/mesh/limit/frame/FK 静态验收
│   ├── amgg_smoke_test.py                # 服务器有限步仿真验收
│   ├── amgg_teleop.py                    # 注册 AMGG 后调用官方遥操程序
│   ├── amgg_record_demos.py              # 注册 AMGG 后调用官方录制程序
│   ├── amgg_convert_hdf5_to_lerobot.py   # HDF5 → LeRobot Dataset v3
│   ├── amgg_teleop_real.py               # 真机遥操入口边界
│   └── amgg_record_real.py               # 真机录制入口边界
├── source/amgg_robot_lab/amgg_robot_lab/
│   ├── assets/                            # 本体 ArticulationCfg 与模型文件
│   ├── contracts/                         # 关节、frame、相机公共 ABI
│   ├── kinematics/                        # URDF FK、Jacobian、DLS IK
│   ├── tasks/                             # 四个场景、MDP、Gym 注册
│   ├── teleop/                            # PICO pipeline、重定向、安全限制
│   ├── recording/                         # HDF5/LeRobot schema
│   └── real/                              # Dry-run 与 ROS 2 后端
└── tests/                                 # 不依赖 Isaac Sim 的单元测试
```

核心文件对应关系：

| 内容 | 文件 |
|---|---|
| 本体导入、驱动增益、初始姿态 | `assets/amgg_robot_cfg.py` |
| 固定关节顺序与限位 | `contracts/amgg_joint_contract.py` |
| base、腕部、TCP 名称 | `contracts/amgg_frame_contract.py` |
| 四相机名称与外参默认值 | `contracts/amgg_camera_contract.py` |
| 离线 FK/Jacobian/IK | `kinematics/amgg_urdf_kinematics.py` |
| Isaac Pink IK 动作 | `tasks/mdp/amgg_actions.py` |
| 成功、失败、进度、观测 | `tasks/mdp/amgg_terms.py` |
| 四个完整环境 | `tasks/amgg_manipulation_env_cfg.py` |
| PICO 控制器映射 | `teleop/amgg_pico_pipeline.py` |
| LeRobot 特征契约 | `recording/amgg_dataset_schema.py` |
| 真机通信 | `real/amgg_ros2_backend.py` |

## 3. 模型做了哪些可追踪修正

原始包位于：

```text
C:\Users\zrobot\Desktop\AM-DPZONGZHUANGURDF20260518
```

生成脚本完成以下确定性操作：

1. 原样保存原始 URDF 和 32 个 STL。
2. 把 `package://.../meshes/...` 改成项目内相对路径。
3. 修复重复的 `ArmR01_Joint`，把第二个关节标准化为 `ArmR02_Joint`。
4. 修复右臂将 link 名误用为 joint 名的问题。
5. 将空格、连字符等字符统一清洗为 USD 安全名称，避免 URDF、USD、Pink 和 ROS 出现不同名字。
6. 将原始全部为 0 的速度上限替换为分组可用上限，并补充 damping/friction。
7. 保留所有原始质量和惯量。
8. 添加 `left_tcp_link`、`right_tcp_link` 和研究用双平行夹爪。

重新生成资产的命令：

```powershell
python amgg_robot_lab\scripts\amgg_prepare_robot_asset.py `
  "C:\Users\zrobot\Desktop\AM-DPZONGZHUANGURDF20260518"
```

服务器通常不需要重新生成，因为标准化 URDF 和 mesh 已经在仓库中；只有更换本体包时才重新运行。

## 4. 公共接口契约

### 4.1 状态和动作

- `observation.state`：23 维，17 个腰部/双臂关节 + 4 个夹爪关节 + 2 个头部关节。
- PICO raw action：18 维，左 TCP 7 + 右 TCP 7 + 4 个夹爪 trigger 值。
- `processed_actions` / 默认 LeRobot `action`：21 维，17 个 IK 关节目标 + 4 个夹爪行程目标。
- 四元数顺序统一为 `x, y, z, w`。
- revolute 使用 rad，prismatic 夹爪使用 m，位置使用 m，速度使用 rad/s 或 m/s。

固定顺序是数据 ABI，不允许依赖解析器偶然返回的 joint 顺序。修改关节顺序意味着数据 schema 版本升级。

### 4.2 四相机

固定数据 key：

```text
observation.images.head
observation.images.left_wrist
observation.images.right_wrist
observation.images.overview
```

仿真默认 640×480、30 Hz。当前外参是初始仿真值，服务器首次启动后必须目视检查视场；真机必须用标定结果替换，不能直接复制仿真默认外参。

### 4.3 FK 和 IK

离线模型直接解析标准化 URDF，支持：

- 任意 link 的 FK；
- 世界表达的 6D geometric Jacobian；
- joint limit 投影；
- 同时约束左右 TCP 的 damped least-squares IK；
- FK → IK → FK 往返测试；
- Jacobian 有限差分测试。

Isaac Lab 运行时使用 Pink IK，但读取同一份 URDF、同一关节顺序、同一限位和同一 TCP。因此离线/真机侧与仿真侧不是两套独立定义。

## 5. 四个论文任务

| Gym Task ID | 任务 | 成功条件 | 失败条件 |
|---|---|---|---|
| `Isaac-AMGG-PickPlace-v0` | 橙色方块放入绿色区域 | 目标 XY/Z 范围内且速度低于阈值 | 掉落、越界、机器人异常、超时 |
| `Isaac-AMGG-BimanualLift-v0` | 双手抬起蓝色长杆 | 高度达标、杆体水平、速度低、左右 TCP 靠近两端 | 掉落、越界、机器人异常、超时 |
| `Isaac-AMGG-Handover-v0` | 从左工作区转移到右目标区 | 黄色圆柱进入右侧目标并稳定 | 掉落、越界、机器人异常、超时 |
| `Isaac-AMGG-Sort-v0` | 红蓝方块放入匹配区域 | 两个物体均颜色匹配且稳定 | 任一物体掉落/越界、机器人异常、超时 |

每个环境都存在名称严格为 `success` 的 termination term。官方 `record_demos.py` 会将它从普通终止项中取出，连续满足 `--num_success_steps` 后标记 episode 成功并写盘；因此无需修改官方 recorder。

reset 时会对任务物体位置和 yaw 做小范围随机化。论文正式实验还应在固定基线跑通后增加光照、材质、质量、摩擦、相机噪声和更宽的物体位姿随机化，并记录随机种子和范围。

## 6. 服务器安装与第一轮验收

在服务器拉取对应分支后：

```bash
cd ~/zzk_data/IsaacLab
./isaaclab.sh -p -m pip install -e amgg_robot_lab/source/amgg_robot_lab
```

先做不启动 Isaac Sim 的检查：

```bash
./isaaclab.sh -p amgg_robot_lab/scripts/amgg_check_robot_asset.py
./isaaclab.sh -p -m pytest amgg_robot_lab/tests -q
```

期望资产检查输出包含：

```text
AMGG robot asset validation passed
links=40, joints=39, actuated_joints=31, meshes=64
```

然后逐个做有限步仿真。第一次建议开窗口检查模型朝向、穿模、桌面高度、相机和初始姿态：

```bash
./isaaclab.sh -p amgg_robot_lab/scripts/amgg_smoke_test.py \
  --task Isaac-AMGG-PickPlace-v0 \
  --num_envs 1 \
  --num_steps 240 \
  --enable_cameras \
  --visualizer kit
```

将 `--task` 依次替换为其余三个 Task ID。验收时重点记录：

- 机器人底盘与地面是否接触而不穿透；
- 初始左右 TCP 是否位于桌面上方；
- 手臂是否自碰撞或持续抖动；
- 夹爪开合方向是否正确；
- 四个相机的方向、遮挡和曝光；
- idle 运行 240 步后是否出现非有限状态或异常 reset；
- Pink 是否报告 frame/joint 名不匹配。

如果第一次 URDF → USD 转换较慢是正常的。若模型导入后物理不稳，优先检查 mesh 碰撞复杂度、惯量、初始穿模和 actuator gain，不要直接放大 solver force 掩盖问题。

## 7. PICO 遥操

以 PickPlace 为例：

```bash
./isaaclab.sh -p amgg_robot_lab/scripts/amgg_teleop.py \
  --task Isaac-AMGG-PickPlace-v0 \
  --visualizer kit \
  --xr \
  --num_envs 1
```

PICO pipeline 使用左右 controller pose，不依赖 hand tracking。`GripperRetargeter` 优先读取 controller trigger，hand pinch 仅作为可选输入。因此出现以下日志并不等于遥操失败：

```text
XDev does not support hand tracking
XR_ERROR_FEATURE_UNSUPPORTED
ControllerTracker initialized (left + right)
```

只要 controller tracker 初始化成功，左右位姿和 trigger 就能工作。第一次需要在调试窗口中核对左右手、TCP 朝向和 recenter；旋转补偿位于 `teleop/amgg_pico_pipeline.py`。

## 8. 自动判定录制

纯遥操入口沿用官方行为，会在 XR 模式临时移除额外传感器相机以保证帧率；这不改变任务与数据集的四相机契约。录制入口则必须显式传入 `--enable_cameras`。先只录制 1 条，确认任务判定、同步性和磁盘写入速度：

```bash
mkdir -p datasets
./isaaclab.sh -p amgg_robot_lab/scripts/amgg_record_demos.py \
  --task Isaac-AMGG-PickPlace-v0 \
  --visualizer kit \
  --xr \
  --enable_cameras \
  --num_demos 1 \
  --num_success_steps 10 \
  --step_hz 30 \
  --dataset_file ./datasets/amgg_pick_place_rgb.hdf5
```

首条验收通过后，再扩大采集数量，例如：

```bash
./isaaclab.sh -p amgg_robot_lab/scripts/amgg_record_demos.py \
  --task Isaac-AMGG-PickPlace-v0 \
  --visualizer kit \
  --xr \
  --enable_cameras \
  --num_demos 10 \
  --num_success_steps 10 \
  --step_hz 30 \
  --dataset_file ./datasets/amgg_pick_place_rgb.hdf5
```

双臂抬杆和分拣建议使用 15 个连续成功 step。HDF5 默认 gzip 压缩，但四路 640×480 RGB 中间文件仍然较大，应先用 1 个 episode 测算磁盘和写入速度，再批量采集。

有效 episode 至少应包含：

```text
actions                         # 18-D PICO/任务空间 raw action
processed_actions               # 21-D 关节位置目标
obs/robot_joint_pos             # 23-D 测量状态
obs/robot_joint_vel
obs/left_tcp_pose
obs/right_tcp_pose
obs/object_state
obs/goal
obs/progress
obs/image_head
obs/image_left_wrist
obs/image_right_wrist
obs/image_overview
```

成功 episode 属性中应有 `success=True`。

## 9. 转换为 LeRobot Dataset v3

建议在独立转换环境安装当前 LeRobot，避免影响 Isaac Sim 的依赖：

```bash
conda create -n amgg_lerobot python=3.12 -y
conda activate amgg_lerobot
pip install lerobot h5py numpy
pip install -e ~/zzk_data/IsaacLab/amgg_robot_lab/source/amgg_robot_lab
```

转换命令：

```bash
python ~/zzk_data/IsaacLab/amgg_robot_lab/scripts/amgg_convert_hdf5_to_lerobot.py \
  ~/zzk_data/IsaacLab/datasets/amgg_pick_place_rgb.hdf5 \
  ~/zzk_data/IsaacLab/datasets/lerobot_amgg_pick_place \
  --task Isaac-AMGG-PickPlace-v0 \
  --repo_id local/amgg_pick_place \
  --fps 30
```

默认行为：

- 只转换 `success=True` 的 episode；
- 使用 21-D `processed_actions` 作为 LeRobot `action`，便于仿真到真机；
- 写入 23-D `observation.state`、14-D TCP pose、任务环境状态和四路视频；
- 每帧写入固定自然语言任务；
- `save_episode()` 后调用 `finalize()`；
- 只生成本地数据集，不自动上传 Hugging Face Hub；
- 在 `meta/amgg_schema.json` 保存 schema、关节名、相机名和源文件信息。

若要训练任务空间策略，可加 `--action_source raw`，此时 `action` 为 18 维。若某次训练只使用低维输入，可在转换阶段加 `--no-include_images`，原始 HDF5 仍保留四路图像以便复现实验。

## 10. 真机接入还需要提供什么

现有 ROS 2 后端使用标准 `sensor_msgs/JointState` 和 `trajectory_msgs/JointTrajectory`，并实现固定顺序重排、状态超时、软件使能门、关节限位、速度限制、跟踪误差和 watchdog。真正接 AMGG 前仍必须提供：

1. 真机 23 个观测关节与 21 个命令关节的精确名称映射；
2. 每个电机的零偏、方向、减速比和编码器单位；
3. 实际软/硬限位、速度、加速度、jerk 和力矩限制；
4. 左右真实夹爪的关节结构、行程、控制单位和反馈；
5. 腕部到真实 TCP 的标定变换；
6. ROS 2 topic、QoS、消息类型，或厂商 SDK 接口；
7. 驱动器 enable/disable、hold、急停和故障清除流程；
8. 控制周期、命令时间戳和 watchdog 要求；
9. 丢包、超时、越限和跟踪误差时的安全动作；
10. 碰撞检测或力矩保护能力；
11. 四个真实相机的型号、序列号、分辨率、FPS 和时间戳来源；
12. 相机内参、畸变、外参和硬件/软件同步方式；
13. PICO 与 robot base 的标定和 clutch/deadman 设计；
14. 一份站立不动的真实 `JointState` 样例；
15. 一份不使能电机也能验证格式的命令样例或仿真控制器。

没有这些信息时，可以完成仿真、数据和 dry-run，但不能负责任地宣称真机运动已安全接通。

## 11. 论文实验建议

- 固定并记录代码 commit、URDF hash、Isaac Lab/Isaac Sim 版本、GPU 和随机种子。
- 数据集按 episode 划分 train/validation/test，禁止按 frame 随机拆分造成泄漏。
- 同时报告 task success rate、completion time、drop/out-of-workspace rate 和人工接管次数。
- 双臂任务额外报告杆体水平误差、双 TCP 末端误差和同步性。
- 报告 raw action、processed action 和 measured state 的延迟及 tracking error。
- 仿真基线稳定后再做 domain randomization 消融，不要一次改变全部变量。
- 真实实验保留失败 episode，但用显式 validity/failure reason 标注，不与成功数据混在一起。
- 每个场景先采 20–50 条 pilot 数据检查动作分布、相机遮挡和成功条件，再扩充数据量。

## 12. 当前验收状态

本机已经完成：

- 原始/标准化 URDF 双份留档；
- 32 个 mesh 引用检查；
- unique link/joint、正速度/力矩限制检查；
- frame、23-D/21-D/18-D 和四相机契约检查；
- FK 有限值检查；
- Jacobian 有限差分测试；
- 双臂 IK 往返测试；
- HDF5 schema 和成功 episode 过滤测试；
- 真机安全 limiter 与 dry-run 后端测试；
- Python compile 检查。

仍需在服务器完成：

- URDF → USD 和 PhysX 首次运行；
- actuator gain 与碰撞稳定性调参；
- 四相机视场确认；
- Pink frame/joint 运行时映射确认；
- 四个任务各至少一次人工成功与自动判定；
- PICO 左右手姿态补偿微调；
- 完整 RGB HDF5 的实际写盘和 LeRobot 视频编码验收。
