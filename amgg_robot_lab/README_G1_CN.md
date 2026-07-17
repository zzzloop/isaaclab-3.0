# Unitree G1 + RH56DFX 三任务实验主线

这条主线冻结了当前有问题的 AMGG 自定义 URDF，直接复用 Isaac Lab 官方
`G1_INSPIRE_FTP_CFG`、Pink 双臂 IK 和 PICO/OpenXR 重定向。自定义内容仅包括任务场景、
自动评测、RH56DFX 数据契约、触觉代理观测和 LeRobot 转换，不修改 Isaac Lab 核心代码。

> 当前仿真手采用官方 G1 Inspire USD。它用每手 12 个运动关节表达 RH56DFX 的耦合连杆；
> 真机 RH56DFX 每手只有 6 个电机。因此仿真内部和真机公共接口必须分层，不能把 24 个
> 仿真手关节原样发送给真机。

## 1. 本体与数据 ABI

| 层 | 状态 | 动作 | 用途 |
|---|---:|---:|---|
| Isaac Lab 仿真 | G1 29 + 手部运动关节 24 = 53D | 双腕位姿 14 + 手部运动关节 24 = 38D | 官方 USD、Pink IK、PICO |
| RH56DFX 真机 | G1 29 + 双手电机 12 = 41D | 双腕位姿 14 + 双手电机 12 = 26D | DDS/真机部署 |
| 力觉 | 双手 12 路，独立字段 | 不并入 action | `observation.tactile` |

RH56DFX 真机电机顺序严格采用 Unitree 官方服务：

```text
right: pinky, ring, middle, index, thumb_bend, thumb_rotation
left:  pinky, ring, middle, index, thumb_bend, thumb_rotation
```

真机位置命令范围为 `[0, 1]`，`0` 表示闭合，`1` 表示张开。仿真 HDF5 当前保留 38D
官方动作，避免在没有真机标定数据时伪造角度到电机归一化量的映射。转换后的 LeRobot
metadata 会明确记录 `hardware_calibration_required=true`。

仿真 `observation.tactile` 是 PhysX 手指接触力聚合得到的 12D 代理量，只能用于接口开发和
消融实验，不能宣称已经复现 RH56DFX 传感器。真机力传感器需要从手部原始协议读取并完成
零偏、量程、方向、延迟和过冲标定；Unitree 公开的 `dfx_inspire_service` DDS 层目前只转发
12 个位置状态。

## 2. 三个任务

| Task ID | 研究维度 | 自动成功条件 | 失败条件 |
|---|---|---|---|
| `Isaac-AMGG-G1-ClutterTransfer-v0` | 杂乱鲁棒性、空间泛化 | 橙色目标块进入绿色区域：XY 误差小于 90 mm、Z 误差小于 55 mm、线速度小于 0.15 m/s | 掉落、越界、非有限状态、关节异常高速、超时 |
| `Isaac-AMGG-G1-BimanualReorient-v0` | 双臂协调、长物体重定向 | 蓝色长杆中心误差小于 80 mm，水平和长轴方向误差小于约 23°，线速度小于 0.15 m/s | 同上 |
| `Isaac-AMGG-G1-PrecisionInsert-v0` | 接触丰富、窄容差插放 | 黄色键块 XY 误差小于 25 mm、Z 误差小于 40 mm、竖直误差小于约 23°、线速度小于 0.10 m/s | 同上 |

成功项统一命名为 `success`，可直接使用官方 `record_demos.py` 的连续成功帧门控。

核心文件：

```text
source/amgg_robot_lab/amgg_robot_lab/
├── contracts/amgg_g1_contract.py               # 仿真/真机/触觉 ABI
├── tasks/amgg_g1_task_specs.py                  # 任务身份与论文描述
├── tasks/amgg_g1_manipulation_env_cfg.py        # 三个 G1 场景和 Manager 配置
├── tasks/mdp/amgg_g1_terms.py                   # 观测、成功和失败条件
└── recording/amgg_g1_dataset_schema.py          # LeRobot schema
scripts/
└── amgg_convert_g1_hdf5_to_lerobot.py           # G1 HDF5 转换器
```

## 3. 服务器安装与离线测试

```bash
cd ~/zzk_data/IsaacLab
git pull
./isaaclab.sh -p -m pip install -e amgg_robot_lab/source/amgg_robot_lab

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
./isaaclab.sh -p -m pytest amgg_robot_lab/tests -q
```

测试覆盖 ABI 维度、Unitree 电机顺序、任务 schema、HDF5 成功过滤与同步性；它不替代
Isaac Sim/PhysX/Pink/OpenXR 运行时验收。

## 4. 按顺序验收三个场景

AMGG 的三个仿真入口会在 Isaac Sim 启动前读取 `nvidia-smi`，默认只暴露物理 0、1、2 号卡，
并优先把物理 2 号卡（第三张卡）映射到 CUDA、Kit 和 CloudXR 的同一个逻辑序号。这样可避开
有问题的物理 3 号卡，并减少跨 GPU 显存复制。启动前建议清除旧的手工设置：

```bash
unset CUDA_VISIBLE_DEVICES NV_GPU_INDEX
```

启动日志应出现 `[AMGG] Preferred physical GPU 2 ...`。临时改用物理 1 号卡可执行
`AMGG_PREFERRED_GPU=1 <启动命令>`；显式传入 `--device` 会关闭自动选择并保留操作者设置。

先跑第一个场景 240 步并查看模型、桌面、相机和接触传感器是否正常：

```bash
./isaaclab.sh -p amgg_robot_lab/scripts/amgg_smoke_test.py \
  --task Isaac-AMGG-G1-ClutterTransfer-v0 \
  --num_envs 1 \
  --num_steps 240 \
  --enable_cameras \
  --visualizer kit
```

依次把 `--task` 换成：

```text
Isaac-AMGG-G1-BimanualReorient-v0
Isaac-AMGG-G1-PrecisionInsert-v0
```

第一次验收必须人工检查：

- G1 是官方固定基座 29DoF 本体和五指手，不再出现旧 AMGG URDF；
- 前视相机覆盖双手和整个任务区，侧视相机无遮挡；
- 手指没有初始穿模，桌面物体静置稳定；
- `action_shape` 为 `(1, 38)`；
- policy keys 包含 `robot_joint_pos`、`rh56dfx_motor_proxy`、`tactile`、两路图像和任务状态；
- 240 步内没有 NaN、Pink frame/joint 错配或连续异常 reset。

## 5. PICO 遥操

先不录制，只测试控制：

```bash
# 任务 1：杂乱物体搬运
./isaaclab.sh -p amgg_robot_lab/scripts/amgg_teleop.py \
  --task Isaac-AMGG-G1-ClutterTransfer-XR-v0 \
  --visualizer kit \
  --xr \
  --num_envs 1

# 任务 2：双臂长杆重定向
./isaaclab.sh -p amgg_robot_lab/scripts/amgg_teleop.py \
  --task Isaac-AMGG-G1-BimanualReorient-XR-v0 \
  --visualizer kit \
  --xr \
  --num_envs 1

# 任务 3：精密插入
./isaaclab.sh -p amgg_robot_lab/scripts/amgg_teleop.py \
  --task Isaac-AMGG-G1-PrecisionInsert-XR-v0 \
  --visualizer kit \
  --xr \
  --num_envs 1
```

纯遥操达到任一成功条件时，终端会输出 `[SUCCESS] Task completed...`，环境自动重置后可继续操作；
失败终止会输出具体 term。纯遥操不保存数据。

`-XR-v0` 任务使用 60 Hz 控制和渲染循环；普通 `-v0` 任务保持原来的 30 Hz 数据验收配置。
该入口直接使用官方 G1 Inspire PICO pipeline。纯遥操模式会由官方脚本暂时移除相机观测，
以降低 XR 延迟；这不改变录制环境的相机 schema。先确认腕部方向、左右手、recenter 和手指
开合，再进入录制。XR 已连接并确认可控后，可把渲染器切换为 `RTX - Minimal`，并把 XR
Render Resolution Multiplier 调到 `0.8`；不要在 XR 建立连接前切换。

## 6. 自动判定并录制 HDF5

`--num_demos 0` 表示持续录制，不会因为成功一条而退出。每次达到成功条件后，当前 episode
自动导出、计数加一并 reset；采集完成时按 `Ctrl+C` 正常结束。建议每次采集使用新的 HDF5 文件名。

```bash
mkdir -p datasets

# 任务 1：杂乱物体搬运，持续录制
./isaaclab.sh -p amgg_robot_lab/scripts/amgg_record_demos.py \
  --task Isaac-AMGG-G1-ClutterTransfer-XR-v0 \
  --visualizer kit \
  --xr \
  --enable_cameras \
  --num_demos 0 \
  --num_success_steps 12 \
  --step_hz 60 \
  --dataset_file ./datasets/amgg_g1_clutter_transfer.hdf5

# 任务 2：双臂长杆重定向，持续录制
./isaaclab.sh -p amgg_robot_lab/scripts/amgg_record_demos.py \
  --task Isaac-AMGG-G1-BimanualReorient-XR-v0 \
  --visualizer kit \
  --xr \
  --enable_cameras \
  --num_demos 0 \
  --num_success_steps 15 \
  --step_hz 60 \
  --dataset_file ./datasets/amgg_g1_bimanual_reorient.hdf5

# 任务 3：精密插入，持续录制
./isaaclab.sh -p amgg_robot_lab/scripts/amgg_record_demos.py \
  --task Isaac-AMGG-G1-PrecisionInsert-XR-v0 \
  --visualizer kit \
  --xr \
  --enable_cameras \
  --num_demos 0 \
  --num_success_steps 15 \
  --step_hz 60 \
  --dataset_file ./datasets/amgg_g1_precision_insert.hdf5
```

录制时单帧满足几何、姿态和速度条件还不算完成；必须连续满足 `--num_success_steps` 帧。
终端出现 `Success condition met! Episode exported; resetting for the next demonstration.` 和递增的
`Recorded N successful demonstrations.` 才表示第 N 条已成功写入，然后程序自动 reset 采集下一条。

一条有效 episode 至少应包含：

```text
actions                              # 官方 G1/PICO 38D
processed_actions                    # Pink 关节目标 38D
obs/robot_joint_pos                  # 仿真完整状态 53D
obs/rh56dfx_motor_proxy              # 12 个真机电机对应的仿真代表关节
obs/tactile                          # 12D PhysX 接触力代理
obs/left_eef_pos, obs/left_eef_quat
obs/right_eef_pos, obs/right_eef_quat
obs/object_state, obs/goal, obs/progress
obs/image_front, obs/image_overview
```

并且 episode 属性中有 `success=True`。

## 7. 转为 LeRobot Dataset v3

转换建议在独立环境运行：

```bash
conda create -n amgg_lerobot python=3.12 -y
conda activate amgg_lerobot
pip install lerobot h5py numpy
pip install -e ~/zzk_data/IsaacLab/amgg_robot_lab/source/amgg_robot_lab
```

示例：

```bash
python ~/zzk_data/IsaacLab/amgg_robot_lab/scripts/amgg_convert_g1_hdf5_to_lerobot.py \
  ~/zzk_data/IsaacLab/datasets/amgg_g1_clutter_transfer.hdf5 \
  ~/zzk_data/IsaacLab/datasets/lerobot_amgg_g1_clutter_transfer \
  --task Isaac-AMGG-G1-ClutterTransfer-v0 \
  --repo_id local/amgg_g1_clutter_transfer \
  --fps 30
```

默认只转换成功 episode、保留两路视频，并把官方 38D task-space raw action 作为 LeRobot
`action`。在 RH56DFX 开合标定完成前，不生成伪造的 26D 真机 action。

## 8. 论文实验协议

三个场景都先做 20–50 条 pilot，修正视角、难度和阈值后再冻结 `v0`。正式实验建议：

1. 按 episode 划分 train/validation/test，禁止按 frame 随机拆分。
2. 固定代码 commit、Isaac Lab/Isaac Sim 版本、GPU、随机种子和任务参数。
3. 每个任务至少报告成功率、完成时间、掉落率、越界率和人工接管次数。
4. Clutter 额外报告目标/干扰物误抓率；Bimanual 额外报告长轴和水平误差；Insert 额外报告
   XY/角度误差、接触峰值和插入耗时。
5. 设置 easy/medium/hard 三档初始位姿和干扰强度，训练分布与测试分布明确分开。
6. 做无触觉、单视角、无随机化、单任务/多任务以及仿真 38D/真机 26D 适配消融。
7. 真机失败 episode 也保留，使用明确的 validity 和 failure reason 标注。

这套场景和协议提供可发表的实验基础，但顶会录用仍取决于方法创新、强基线、公平消融、
真实机器人结果和统计显著性，不能仅靠三个场景保证。

## 9. 真机适配前还需采集的信息

在写 RH56DFX DDS/串口后端之前，需要提供：

1. 左右手铭牌完整型号（`RH56DFX-2L/2R` 或带腕版本）和固件版本；
2. 实验室正在使用的 `dfx_inspire_service` commit、启动参数和网卡；
3. 静止、全开、全闭三组 `rt/inspire/state` 样例；
4. 每个电机安全速度、力阈值和允许命令频率；
5. 力传感器读取方式、原始 12 路样例、零偏和单位；
6. G1 上肢控制接口、模式切换、急停、watchdog 和碰撞保护流程；
7. 真机相机数量、分辨率、FPS、时间戳源、内外参和同步方式。

拿到这些内容后，再实现 24 个仿真运动关节与 12 个真实电机之间的标定映射，以及仿真
接触力到真实传感器分布的归一化/噪声/延迟模型。

## 10. 官方依据

- Isaac Lab 官方 G1 任务：`Isaac-PickPlace-G1-InspireFTP-Abs-v0`
- Unitree RH56DFX 服务：<https://github.com/unitreerobotics/dfx_inspire_service>
- RH56DFX 产品参数：<https://inspire-robots.com/dexterous%20hands/rh56dfx-series/>
