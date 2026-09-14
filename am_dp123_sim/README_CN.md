# AM-DP123 二代机器人 Pico 仿真验证

本目录是独立新增入口，不依赖已有 AMGG 任务注册，也不连接电机控制接口。
二代模型使用原始 URDF 的关节原点、轴向、限位和网格，双臂各 7 自由度，夹爪共 4 个可动关节。
当前范围是固定底座的双臂和夹爪动作验证；底盘、腰部、头部保持初始关节角。

## 启动

所有命令从 IsaacLab 仓库根目录执行。使用你已有的 Isaac Lab / Isaac Sim 环境，不需要安装本目录。
网格由 Git LFS 管理，拉取代码后运行 `git lfs pull`，避免拿到文本指针而非 STL。

终端一：

```bash
./isaaclab.sh -p am_dp123_sim/run_sim.py --mode kinematic
```

终端二：使用你原来能接收 Pico 话题的 ROS1 环境，启动现有 Pico 位姿发布端，然后运行：

```bash
python am_dp123_sim/ros1_pico_bridge.py --host 127.0.0.1
```

桥接器只订阅以下五个原始话题，不运行一代 IK，不发布任何机器人控制话题：

| 话题 | 类型 |
| --- | --- |
| `/teleop/pose/hmd` | `geometry_msgs/PoseStamped` |
| `/teleop/pose/left_controller` | `geometry_msgs/PoseStamped` |
| `/teleop/pose/right_controller` | `geometry_msgs/PoseStamped` |
| `/teleop/controller/left_joy` | `sensor_msgs/Joy` |
| `/teleop/controller/right_joy` | `sensor_msgs/Joy` |

Joy 约定与参考工程一致：`axes[0]` 为 0–1 扳机值，`buttons[2]` 为 A，`buttons[3]` 为 B。
ROS Python 与 Isaac Sim Python 分进程，通过 UDP 15051 通信。
如果两个进程分布在不同机器，桥接器的 `--host` 指向仿真机器，仿真入口使用 `--bind 仿真机器局域网IP`。
默认仅监听本机；网络输入没有身份认证，应只用于可信实验网络。
不需要启动一代硬件控制或一代 IK 节点。

## 使用与标定

1. 等待模型和五个话题就绪，先松开所有按键。
2. 面向期望的正前方，自然放置双手，同时按下双 A。当前手柄姿态与机器人当前末端姿态建立对应关系，启用瞬间不跳变。
3. 小幅移动单侧手柄，分别验证前后、左右、上下；再单独转动手腕。场景中的红、绿、蓝坐标轴分别表示末端目标的 X、Y、Z 方向。
4. 双 B 暂停。调整站位后松开 A，再按双 A，重新标定后继续。
5. 断流超过 0.5 秒会暂停。恢复数据后仍需松开并重新按双 A，不会自动继续执行。

坐标转换沿用 `ROS_v2/new_teleop_pico_ws` 的预处理约定：Pico X右/Y上/Z前，转换为机器人 X前/Y左/Z上。
旋转使用 `P R P^-1`，不能只交换四元数分量。手柄平移先减去头显平移；手腕姿态保持绝对旋转语义，再按启用时的朝向和末端姿态标定。
位置默认 1:1，不继承一代机器人的臂长、IK 参数或末端固定偏置。
这是适配二代 URDF 的相对标定控制，不是逐关节复现一代真机指令；尚不能声称与未提供的二代真机 IK 输出完全一致。

`teleop_config.json` 可配置位置缩放、关节速度上限、IK 接受误差和初始关节角。
`home` 按 URDF 原始关节名填写，单位 rad；默认腰部、头部为零，双肘 0.3 rad。
每个夹爪两指默认从 `[0, 0]` 到 `[-0.32, 0.32]` rad，对应 URDF 限位端点。
请在画面中确认实际开合方向；如与模型设计相反，交换 `gripper_open` 与 `gripper_closed`。端点方向未经过真机标定。

## 验证模式与输出

- `kinematic`：每步直接写入经过限位和限速的关节位置，便于排查模型和映射。不是动力学验证。
- `servo`：使用位置执行器跟踪目标，便于观察跟踪延迟。增益是仿真默认值，尚未根据真机辨识。

两种模式都固定底座、关闭机器人重力和自碰撞；不验证平衡、接触安全、承载或电机力矩真实性。
画面显示在 Isaac Sim 窗口；本入口不向 Pico 头显回传图像。

```bash
./isaaclab.sh -p am_dp123_sim/run_sim.py --mode servo
```

IK 对每侧双臂单独求解，保留其他关节。任一侧不可达或超出误差阈值时，两臂同时保持前次命令，并显示 `IK REJECTED / HOLD`。
默认接受阈值为 0.03 m、0.2 rad，可在配置中收紧；这是求解接受标准，不是已经达到的实测精度。

运行记录保存在 `am_dp123_sim/outputs/`：

- `*_pico.jsonl`：接收的原始位姿、按键及相对接收时间。
- `*_metrics.csv`：每 0.5 秒记录左右臂目标误差、关节跟踪误差，以及独立 URDF FK 与仿真实际连杆姿态的差异。

`target_*` 偏大而 `fk_import_*` 很小，优先检查不可达目标、限速与执行器跟踪。
`fk_import_*` 明显非零，优先检查 URDF 导入、关节映射、基座高度及坐标约定。
暂停时目标坐标轴保留最后位置，误差可能非零，需结合 `status` 判断。

## 离线测试与回放

```bash
./isaaclab.sh -p am_dp123_sim/test_offline.py
./isaaclab.sh -p am_dp123_sim/run_sim.py --headless --steps 120
./isaaclab.sh -p am_dp123_sim/replay_pico.py am_dp123_sim/outputs/你的记录_pico.jsonl
```

回放前重新启动仿真以恢复相同初始姿态，停止实时桥接器；回放保留原始双 A/B 事件。
改变回放速度会影响限速、断流阈值和结果，不保证动力学轨迹逐帧相同。
离线测试覆盖 61 个网格、31 个可动关节、惯性正定性、有限差分雅可比、可达/不可达 IK、Pico 方向、暂停恢复和 UDP 异常帧。
已通过离线测试；本地未启动 Isaac Sim 或连接 Pico，运行时渲染及设备联调仍待验证。

## 资源与版本

原始资源来自 `C:\Users\zrobot\Desktop\AM-DP123`。网格字节保持不变；URDF 仅规范化文本空白，所有结构与参数保持不变。
`assets/manifest.json` 保存 SHA-256；运行时只在 `.cache/` 生成绝对网格路径的 URDF 和 USD。
缓存与运行日志不提交，代码不依赖原始 Windows 路径。

使用本项目 3.0 beta2 API：ProxyArray 的 `.torch`、`*_index` 写入接口，以及 xyzw 四元数。
项目基线为 `00aa9cc60c3ef680b381d9e081ffc437b92dd53d`（`amgg-recording-recovery`）。
官方更新目标为 `0603cb1710dcda13087665f63abf9ac483b63c05`（`release/3.0.0-beta2`）。
不合并官方 2.x `main` 到此 3.0 项目。

参考：
- 一代话题与按键：`F:\ROS_v2\new_teleop_pico_ws\src\humanoid_teleop\src\humanoid_teleop\core\realtime_reader.py`
- 一代坐标约定：同目录 `preprocessor.py`
- [官方 beta2 分支](https://github.com/isaac-sim/IsaacLab/tree/release/3.0.0-beta2)
