# AMGG Robot Lab 配置与操作说明

本文说明 AMGG 自定义本体、场景、相机、PICO 遥操、示范数据录制、LeRobot 转换及真机接口的
放置位置、配置顺序和操作方法。

当前项目处于“完整骨架已建立，等待真实本体资料”的阶段。所有依赖机器人真实参数的入口都会
明确报出未配置错误，不会使用虚假的关节、运动学或真机参数。

## 1. 项目边界

AMGG 代码与 Isaac Lab 核心代码隔离：

```text
PICO / OpenXR
    ↓
amgg_pico_pipeline + amgg_retargeter + amgg_safety
    ↓
固定顺序的关节位置指令 q_cmd
    ├── Isaac Lab 仿真任务
    └── 真机 ROS 2 / 厂商 SDK 后端
              ↓
      qpos + q_cmd + cameras
              ↓
       HDF5 → LeRobot
```

仿真与真机应共用以下接口：

- 相同的关节名称和顺序；
- 相同的单位，关节使用 rad，位置使用 m；
- 相同的左右手、夹爪和相机语义；
- 相同的 `observation.state` 与 `action` 数据定义；
- 相同的时间戳和安全限制原则。

## 2. 目录说明

```text
amgg_robot_lab/
├── README.md
├── README_CN.md
├── pyproject.toml
├── scripts/
│   ├── amgg_check_robot_asset.py
│   ├── amgg_teleop_real.py
│   ├── amgg_record_real.py
│   └── amgg_convert_hdf5_to_lerobot.py
├── source/amgg_robot_lab/
│   ├── config/extension.toml
│   ├── pyproject.toml
│   ├── setup.py
│   └── amgg_robot_lab/
│       ├── assets/
│       ├── contracts/
│       ├── kinematics/
│       ├── tasks/
│       ├── teleop/
│       ├── recording/
│       └── real/
└── tests/
```

### 2.1 本体模型

机器人模型入口：

```text
source/amgg_robot_lab/amgg_robot_lab/assets/amgg_robot_cfg.py
```

URDF 固定放置为：

```text
source/amgg_robot_lab/amgg_robot_lab/assets/data/urdf/amgg_robot.urdf
```

网格放置在：

```text
source/amgg_robot_lab/amgg_robot_lab/assets/data/meshes/
```

建议保留 URDF 原来的 meshes 子目录结构。URDF 中的 mesh 路径应满足以下一种形式：

```text
../meshes/visual/xxx.stl
../meshes/collision/xxx.stl
package://amgg_description/meshes/xxx.dae
```

若使用 `package://`，还需要提供 ROS package 名称与真实目录映射。不要在收到模型前手工修改
mesh 路径，以免破坏真机与仿真共用的机器人描述。

`amgg_robot_cfg.py` 后续负责：

- URDF 导入及 USD 生成/引用；
- articulation root；
- 固定、轮式或浮动基座；
- 初始姿态；
- actuator 分组；
- stiffness、damping、armature；
- 速度、力矩和关节位置限制；
- 自碰撞、重力和求解器参数。

### 2.2 场景和任务物体

场景资产放置在：

```text
source/amgg_robot_lab/amgg_robot_lab/assets/data/scenes/
```

可操作物体放置在：

```text
source/amgg_robot_lab/amgg_robot_lab/assets/data/objects/
```

每个场景或物体至少需要说明：

- 文件单位和坐标系；
- 初始位置与四元数顺序；
- visual 与 collision 网格；
- 物体质量和惯量；
- 是否固定；
- 任务成功区域或目标对象；
- reset 时是否随机化。

仿真任务总配置位于：

```text
source/amgg_robot_lab/amgg_robot_lab/tasks/amgg_pick_place/amgg_env_cfg.py
```

MDP 逻辑位于：

```text
tasks/amgg_pick_place/mdp/amgg_observations.py
tasks/amgg_pick_place/mdp/amgg_events.py
tasks/amgg_pick_place/mdp/amgg_terminations.py
```

它们分别负责观测、重置/随机化、成功/失败判定。任务注册完成后的 ID 为：

```text
AMGG-PickPlace-v0
```

### 2.3 关节契约

固定关节顺序定义在：

```text
source/amgg_robot_lab/amgg_robot_lab/contracts/amgg_joint_contract.py
```

需要填充：

```python
AMGG_JOINT_SPECS
AMGG_CONTROLLED_JOINT_NAMES
AMGG_OBSERVED_JOINT_NAMES
```

每个关节必须明确：

```text
name
group
lower_limit_rad
upper_limit_rad
home_position_rad
max_velocity_rad_s
max_effort_nm
command_enabled
```

关节顺序是公开 ABI，不能依赖 Isaac Lab、URDF 解析器或 ROS 2 返回的偶然顺序。以下组件都必须
显式映射到同一顺序：

- Isaac Lab articulation joint order；
- FK/IK 输入输出；
- PICO 遥操输出；
- 真机 command topic；
- 真机 joint state；
- HDF5；
- LeRobot。

### 2.4 Link、TCP 和坐标系

关键 frame 定义在：

```text
source/amgg_robot_lab/amgg_robot_lab/contracts/amgg_frame_contract.py
```

需要提供：

```text
base_link
torso_link
left_wrist_link
right_wrist_link
left_tcp_link
right_tcp_link
```

如果 URDF 没有独立 TCP link，需要额外提供腕部到 TCP 的刚性变换：

```text
translation: x y z [m]
quaternion: x y z w
```

还需要明确：

- PICO 世界坐标系；
- Isaac Lab 世界坐标系；
- 机器人 base 坐标系；
- 左右手腕坐标方向；
- 四元数顺序；
- 左右手是否需要额外旋转补偿。

### 2.5 相机

相机契约位于：

```text
source/amgg_robot_lab/amgg_robot_lab/contracts/amgg_camera_contract.py
```

每个相机需要配置：

```text
name
parent_link
width_px
height_px
fps
translation_m
quaternion_xyzw
```

相机名会直接形成数据集 key：

```text
observation.images.<camera_name>
```

开始正式采集后，不应随意修改相机数量、名称、分辨率和语义。仿真图像与真机图像必须使用相同
key；分辨率可以通过训练预处理统一，但必须记录原始配置。

## 3. FK 与 IK

FK 入口：

```text
source/amgg_robot_lab/amgg_robot_lab/kinematics/amgg_fk.py
```

IK 入口：

```text
source/amgg_robot_lab/amgg_robot_lab/kinematics/amgg_ik.py
```

拿到模型后按以下顺序实现：

1. 解析 URDF joint tree、axis、origin 和 limit；
2. 核对左右臂运动链及腰部公共链；
3. 定义腕部和 TCP；
4. 实现单点 FK；
5. 用已知零位和典型姿态核验 FK；
6. 实现带 seed 的双臂 IK；
7. 加入关节限位、连续解和奇异点处理；
8. 根据需要加入自碰撞和场景碰撞；
9. 做 FK → IK → FK 往返误差测试；
10. 在 Isaac Lab 中对比目标 TCP 和实际 TCP。

用于真机前还必须核对真实零偏、编码器方向、减速比和 TCP 标定。URDF 几何正确不等于真机零位
一定正确。

## 4. PICO 遥操

PICO pipeline：

```text
source/amgg_robot_lab/amgg_robot_lab/teleop/amgg_pico_pipeline.py
```

机器人重定向：

```text
source/amgg_robot_lab/amgg_robot_lab/teleop/amgg_retargeter.py
```

公共安全检查：

```text
source/amgg_robot_lab/amgg_robot_lab/teleop/amgg_safety.py
```

推荐数据流：

```text
PICO 左右腕/手指
→ 坐标标定
→ workspace clamp
→ 双臂 IK
→ 关节位置、速度限制
→ AMGG canonical q_cmd
→ 仿真或真机后端
```

正式启用真机前必须加入：

- START/STOP/RESET；
- deadman 或 clutch；
- 跟踪有效性检查；
- 指令时间戳和 sequence；
- watchdog；
- 关节位置、速度和加速度限制；
- 工作空间限制；
- 急停；
- 通信丢失后的安全行为。

## 5. 数据录制与 LeRobot

数据接口位于：

```text
source/amgg_robot_lab/amgg_robot_lab/recording/amgg_dataset_schema.py
```

Isaac Lab recorder 扩展位于：

```text
recording/amgg_recorder_terms.py
recording/amgg_recorder_cfg.py
```

建议固定字段：

```text
observation.state = 实际测量 qpos，float32[N]
action            = 实际发送 q_cmd，float32[M]
timestamp         = 单调时间戳
task              = 语言任务描述
observation.images.<name> = 同一控制帧对应的图像
```

不要用测量状态代替 action。action 必须是实际发送给仿真或真机的命令。原始 PICO 位姿可以作为
辅助调试字段保存，但不应覆盖真实关节命令。

LeRobot 转换入口：

```text
scripts/amgg_convert_hdf5_to_lerobot.py
```

转换器将在最终关节和相机 schema 冻结后实现。

## 6. 真机后端

硬件无关接口：

```text
source/amgg_robot_lab/amgg_robot_lab/real/amgg_robot_backend.py
```

ROS 2 占位实现：

```text
source/amgg_robot_lab/amgg_robot_lab/real/amgg_ros2_backend.py
```

真机接口至少需要实现：

```text
connect()
enable()
read_state()
send_joint_position_targets()
stop()
disconnect()
```

如果使用厂商 SDK 而不是 ROS 2，应新建 `amgg_vendor_backend.py`，实现同一接口，不要把 SDK 调用
写入 PICO pipeline 或数据录制器。

## 7. 服务器安装

external project 应放在 Isaac Lab 仓库外：

```text
~/zzk_data/
├── IsaacLab/
└── amgg_robot_lab/
```

安装：

```bash
cd ~/zzk_data/amgg_robot_lab
../IsaacLab/isaaclab.sh -p -m pip install -e source/amgg_robot_lab
```

验证纯 Python 契约：

```bash
../IsaacLab/isaaclab.sh -p -m unittest discover -s tests -v
```

检查模型入口：

```bash
../IsaacLab/isaaclab.sh -p scripts/amgg_check_robot_asset.py
```

在收到并实现 URDF 解析前，模型检查会明确提示尚未配置，这是预期行为。

## 8. 完成本体配置后的仿真命令

任务注册和环境配置完成后，先进行无 PICO 的资产与 idle-action 验证，再运行遥操：

```bash
cd ~/zzk_data/amgg_robot_lab

../IsaacLab/isaaclab.sh -p \
    ../IsaacLab/scripts/environments/teleoperation/teleop_se3_agent.py \
    --task AMGG-PickPlace-v0 \
    --visualizer kit \
    --xr \
    --num_envs 1
```

录制仿真示范：

```bash
mkdir -p datasets

../IsaacLab/isaaclab.sh -p \
    ../IsaacLab/scripts/tools/record_demos.py \
    --task AMGG-PickPlace-v0 \
    --visualizer kit \
    --xr \
    --num_demos 1 \
    --num_success_steps 10 \
    --step_hz 30 \
    --dataset_file ./datasets/amgg_baseline.hdf5
```

存在相机配置后再添加：

```text
--enable_cameras
```

## 9. 真机操作顺序

不要从 PICO 直接跳到真机。应依次完成：

1. URDF 静态检查；
2. Isaac Lab 本体加载；
3. idle pose 稳定性；
4. 单关节小幅运动；
5. FK 检查；
6. 仿真 IK；
7. 仿真 PICO 遥操；
8. 真机只读状态；
9. 真机低速单关节；
10. 真机 shadow mode；
11. 带 deadman 的低速 PICO；
12. 真机数据录制。

`scripts/amgg_teleop_real.py` 和 `scripts/amgg_record_real.py` 在真机后端、安全限制和急停方式
明确之前保持禁用。

## 10. 需要提供的资料

### 第一批：仿真本体和运动学

```text
amgg_robot.urdf
meshes/ 完整目录
固定/轮式/浮动基座说明
控制关节顺序
真实零位与方向
base/torso/左右腕/TCP link 名称
夹爪或灵巧手控制说明
```

### 第二批：自定义场景和相机

```text
场景模型
操作物体模型
初始位姿
成功条件
相机数量、名称、分辨率、FPS
相机内参和外参
```

### 第三批：真机

```text
ROS 2 版本或厂商 SDK
command/state topic 或 API
消息定义
控制频率
网络拓扑
上电、使能、停止和急停流程
关节位置/速度/力矩安全限制
相机数据接口和时间戳来源
```

## 11. 当前有意保持未实现的入口

以下入口只有在真实资料齐全后才会实现：

- `get_amgg_robot_cfg()`；
- `compute_amgg_forward_kinematics()`；
- `solve_amgg_inverse_kinematics()`；
- `build_amgg_env_cfg()`；
- `build_amgg_pico_pipeline()`；
- AMGG recorder terms；
- 真机 ROS 2/厂商 SDK 后端；
- HDF5 → LeRobot 转换器。

这些入口当前主动报错是安全设计，不是遗漏。
