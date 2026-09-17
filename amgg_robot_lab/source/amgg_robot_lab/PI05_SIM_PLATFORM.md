# AM-DP123 PI0.5 仿真验证平台使用说明

本文档面向 Ubuntu 22.04、Isaac Sim 6.1、Isaac Lab 3.0 和 `isaaclab30` 环境。平台只使用 AM-DP123
URDF、四路仿真相机和 18 维关节位置控制，不经过 PICO、Pink IK 或真机通信。

## 1. 更新代码与进入环境

```bash
cd ~/zzk_data/IsaacLab
git switch release/3.0.0
git pull --ff-only zzzloop release/3.0.0

conda activate isaaclab30
unset VIRTUAL_ENV PYTHONPATH ISAAC_PATH CARB_APP_PATH EXP_PATH LD_PRELOAD
export ISAACLAB_CXR_ACCEPT_EULA=yes
```

下面所有命令都从 `~/zzk_data/IsaacLab` 执行。`--viz none` 和 `--viz viser` 都不依赖服务器显示器；
显示器接在 GPU2 不需要修改命令。`--device cuda:0` 只选择计算 GPU，可按实际负载改为 `cuda:1`、
`cuda:2` 或 `cuda:3`。

本文优先给出 Isaac Lab 3.0 的 `uv run` 命令。如果服务器继续复用已经通过 `./isaaclab.sh -i` 安装好的
`isaaclab30` conda 环境，并且 `uv run` 开始创建新的 `.venv`，可把下文 IsaacLab 客户端命令开头的
`uv run python` 或 `uv run --extra viser python` 原样替换为 `./isaaclab.sh -p`。OpenPI 服务端自身的
`uv run scripts/serve_policy.py` 不要替换。使用 conda 兼容命令前确认：

```bash
./isaaclab.sh -p -c "import isaacsim, viser; print('isaaclab30 runtime ok')"
```

## 2. 无头链路冒烟

先运行保持策略，验证 URDF、物理环境、四路相机、23 维状态、动作适配和 episode 写盘：

```bash
uv run python amgg_robot_lab/scripts/am_dp123_pi05_eval.py \
    --policy mock_hold \
    --max_steps 120 \
    --viz none \
    --device cuda:0 \
    --kit_args "--/renderer/multiGpu/enabled=false"
```

成功时终端依次出现：

```text
[pi05] starting ...
[pi05] environment ready
[pi05] OpenPI payload: ...
[pi05] wrote episode_000000 ...
[pi05] finished after 120 steps ...
```

输出目录为 `outputs/am_dp123_pi05_eval/<timestamp>/`。

## 3. SSH 下观察机器人动作

服务器使用 Viser，无需 X11：

```bash
uv run --extra viser python amgg_robot_lab/scripts/am_dp123_pi05_eval.py \
    --policy mock_sine \
    --max_steps 600 \
    --viz viser \
    --device cuda:0 \
    --record_images \
    --image_stride 15 \
    --kit_args "--/renderer/multiGpu/enabled=false"
```

本地电脑建立 SSH 隧道：

```bash
ssh -L 8080:127.0.0.1:8080 kemove@<服务器IP>
```

浏览器打开 `http://127.0.0.1:8080`。机器人左右肩肘应平滑小幅运动，夹爪与其他关节保持初始位置。
`--extra viser` 会使用仓库声明的官方 Viser 可选依赖；不要单独执行未锁版本的 `pip install viser`，也不要因此
改用需要本地窗口的 Kit 模式。

## 4. 验收记录

把 `<timestamp>` 换成实际目录：

```bash
uv run python amgg_robot_lab/scripts/am_dp123_pi05_episode.py \
    --mode validate \
    --episode outputs/am_dp123_pi05_eval/<timestamp>/episode_000000.npz
```

验收会检查：

- 23 维位置/速度、3 维物体位置、18 维执行目标及模型动作 shape；
- NaN、Inf 和执行关节限位；
- `inference_valid` 与 `inference_error` 是否一致；
- JSON 中动作/状态 ABI、步数、模型动作维度和格式版本是否与 NPZ 一致。

输出包含 `valid` 才表示该 episode 通过离线验收。

## 5. 回放记录

```bash
uv run --extra viser python amgg_robot_lab/scripts/am_dp123_pi05_episode.py \
    --mode replay \
    --episode outputs/am_dp123_pi05_eval/<timestamp>/episode_000000.npz \
    --viz viser \
    --device cuda:0 \
    --kit_args "--/renderer/multiGpu/enabled=false"
```

可加 `--max_steps 300` 只回放前 300 步。回放使用记录中的 `applied_joint_target`，不会调用策略服务；
它用于肉眼检查动作是否出现跳变、变形或异常关节姿态。

## 6. 连接真实 PI0.5 服务

OpenPI 服务端与 IsaacLab 客户端使用两个独立进程和环境。

IsaacLab 客户端需要 OpenPI 仓库中的轻量客户端。Isaac Lab 3.0 要求 NumPy 2，而 OpenPI 客户端当前的
包元数据仍声明 `numpy<2`，因此不要在 `isaaclab30` 中直接安装其依赖并降级 NumPy。使用客户端源码路径，
并先验证导入：

```bash
export PYTHONPATH="$HOME/openpi/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
uv run python -c "from openpi_client import image_tools, websocket_client_policy; print('openpi client ok')"
```

服务端：

```bash
conda activate openpi
cd ~/openpi

uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=<AM_DP123_CONFIG> \
    --policy.dir=<CHECKPOINT_PATH>
```

IsaacLab 客户端：

```bash
conda activate isaaclab30
cd ~/zzk_data/IsaacLab
export PYTHONPATH="$HOME/openpi/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"

uv run --extra viser python amgg_robot_lab/scripts/am_dp123_pi05_eval.py \
    --policy remote \
    --host 127.0.0.1 \
    --port 8000 \
    --prompt "pick up the orange cube and place it on the green target" \
    --action_layout amgg_robot_lab/source/amgg_robot_lab/amgg_robot_lab/policy/layouts/<CONFIRMED_LAYOUT>.json \
    --action_horizon 8 \
    --max_steps 1800 \
    --viz viser \
    --device cuda:0 \
    --record_images \
    --image_stride 15 \
    --kit_args "--/renderer/multiGpu/enabled=false"
```

当前 `am_dp123_pi05_32_template.json` 是未确认模板，`confirmed=false` 且 `source_indices=null`，不能运行。
必须先确认 PI0.5 的 32 维动作含义，填写 18 个执行关节对应的 `source_indices`，设置正确的
`absolute_joint_position` 或 `delta_joint_position`、scale、offset，并将 `confirmed` 改为 `true`。

## 7. 安全与数据语义

- 所有执行目标经过关节位置限位和逐 step 速度限位。
- reset 后的速度限位从实测关节位置开始。
- 观测、图像处理、WebSocket、返回值或动作适配失败时保持上一安全目标。
- 连续 5 次失败后退出，避免无限重试。
- 失败 step 的 `model_action` 是零占位，必须结合 `inference_valid=false` 和 `inference_error` 解读。
- episode 终止行使用 reset 前的 `final_obs`，不会混入下一 episode 的初始状态。
- 仿真相机参数不是实际相机标定结果；真机部署前仍需统一相机标定、状态/动作归一化、控制频率和延迟。

OpenPI 客户端安装与远程推理接口参考：
<https://github.com/Physical-Intelligence/openpi/blob/main/docs/remote_inference.md>。
