# FlowMatch：FR3 100k策略

此目录就是新增的LeRobot策略：**`src/lerobot/policies/flowmatch/`**。本分支仅新增此目录，不修改factory、其他策略、数据集代码、项目依赖或锁文件。参考本项目ACT/Diffusion/ACMT的config/policy/processor接口，使用已有100k SigLIP + 条件流动作模型；不是官方π0权重。

## 主要文件

| 文件 | 作用 |
|---|---|
| `configuration_flowmatch.py` | `FlowMatchConfig`，注册类型`flowmatch`，声明输入输出特征 |
| `modeling_flowmatch.py` | `FlowMatchPolicy(PreTrainedPolicy)`，加载权重、生成动作、前4步队列/reset |
| `processor_flowmatch.py` | 原生LeRobot前后处理格式，避免重复归一化 |
| `_core/` | 保留训练结构和参数名称的SigLIP、动作流模型 |
| `runtime.py` | 真正独立加载vision.pt/action.pt的视觉/动作worker |
| `__main__.py`、`cli.py` | 先注册flowmatch，再调用原有LeRobot入口；不改原入口源码 |
| `export.py`、`infer_saved.py` | 转换原生格式、从保存观测离线推理 |
| `download_weights.py`、`assets/weights_manifest.json` | 下载100k权重并核对SHA256 |
| `deployment.example.json` | 现场相机、关节和夹爪映射模板，默认未填写 |
| `tests/`、`verify_100k.py`、`validation.json` | 策略接口测试、实权重数值复核与记录 |

## 输入输出

- 四路顺序固定为side/top/wrist1/wrist2，各480×640 RGB，uint8或float[0,1]。
- state为实测FR3 q1…q7（rad）+实测gPO/255，8维。
- 原始模型输出16×8：GELLO绝对关节目标（rad）+连续raw夹爪命令；默认执行前4步。
- `predict_raw_action_chunk`用于离线原始动作；`predict_action_chunk/select_action`转换为FR3 q[7]+目标gPO/255，要求先填写物理映射。
- task是checkpoint固定词表ID，不是自由语言模型。原生pipeline用精确`task`文本或配置中的`default_task_id`；直接离线调用也支持`task_id`。
- state/action使用checkpoint统计各归一化一次。图像明确使用官方SigLIP **PIL后端**，不使用新版默认Torchvision缩放，不套ACMT裁剪/残差恢复/夹爪processor。

## 1. 注册与启动：只加目录时必须注意

LeRobot不会仅因出现新文件夹就自动扫描策略。**本目录导入时注册config，之后原工厂已有的动态fallback即可找到模型和processor**，所以不用修改factory或父目录`__init__.py`。

自有Python启动脚本先导入一次：

```python
from lerobot.policies.flowmatch import FlowMatchConfig, FlowMatchPolicy
from lerobot.policies.factory import get_policy_class, make_policy_config

assert get_policy_class("flowmatch") is FlowMatchPolicy
config = make_policy_config("flowmatch", device="cpu")
# 实际推理应加载下节导出的原生目录，不能直接用随机初始化config。
```

若使用原项目录制/推理命令，换成本目录的启动前缀，其余参数沿用对方已经工作的命令：

```bash
python -m lerobot.policies.flowmatch record <原来的录制/推理参数>
python -m lerobot.policies.flowmatch eval <原来的评估参数>
```

该入口只做注册与参数转交；`record/eval`仍执行原脚本，`record`会按原命令连接硬件，只应在现场调试时运行。本次测试仅检查参数转交，没有执行这些硬件入口。**仍直接执行旧`lerobot-record`且没有预先导入flowmatch，会无法识别新类型**。这里没有修改旧入口；对方可使用上面的前缀，或在自己的启动脚本中加一次导入。

## 2. 环境与100k权重

部署端沿用该FR3项目的Linux/Python≥3.12环境。若新建环境，先从项目根可编辑安装已有extras，再安装本目录额外依赖：

```bash
uv pip install -e '.[dataset,fr3,transformers-dep,diffusers-dep]' -r src/lerobot/policies/flowmatch/requirements-inference.txt
```

原`record/eval`所需的硬件/显示/采集依赖仍按原项目部署流程安装。本目录不改主锁文件；若uv自动同步回其他依赖版本，可在已装好模型依赖的环境中用明确的Python解释器，或`uv run --no-sync`执行。GPU版Torch沿用目标机已经验证的驱动匹配环境；本次数值验证使用Torch2.7.1+cpu。

已有原始权重就直接使用。需下载时：

```bash
gh auth login
python src/lerobot/policies/flowmatch/download_weights.py --output outputs/flowmatch_100k/source_weights
```

来源是[zmy513/fr3-siglip-flow的100k Release](https://github.com/zmy513/fr3-siglip-flow/releases/tag/v0.1.0-100k)，私有仓库需访问权限。下载器可用`--source-dir`读取本地文件，校验哈希并拒绝覆盖不同内容。

- `policy_100k.pt`：去优化器的整体推理checkpoint，约389MiB。
- `components/vision.pt`、`components/action.pt`：独立视觉/动作权重，约355MiB/35MiB。
- `step_0100000.pt`：含优化器的训练checkpoint，推理不需要。

权重不进入Git。重命名策略不改变100k参数；两组件必须属于同一bundle。恢复训练权重不需要另下载初始SigLIP。

## 3. 导出与离线读取

整体转换（输出必须是新目录）：

```bash
python -m lerobot.policies.flowmatch export --checkpoint outputs/flowmatch_100k/source_weights/policy_100k.pt --output outputs/flowmatch_100k/whole --device cpu
```

目录内有标准`config.json/model.safetensors/policy_preprocessor.json/policy_postprocessor.json`，config类型为`flowmatch`。旧版`fr3split`原生目录请用原始100k PT重新导出，不直接沿用旧type字段。

读取标准目录：

```python
from lerobot.policies.flowmatch import FlowMatchPolicy

policy = FlowMatchPolicy.from_pretrained("outputs/flowmatch_100k/whole", local_files_only=True)
raw_chunk = policy.predict_raw_action_chunk(batch, seed=101)  # [B,16,8]，供离线分析
# 映射配置核实后再使用policy.select_action(batch)，返回[B,8]的FR3目标。
```

从保存观测推理：NPZ包含`rgb_side/rgb_top/rgb_wrist1/rgb_wrist2`（HWC RGB uint8）、`state`（8维实测状态）、`task_id`（整数）。

```bash
python -m lerobot.policies.flowmatch infer --policy-path outputs/flowmatch_100k/whole --observation observation.npz --output outputs/flowmatch_prediction.npz --device cpu --seed 101
```

## 4. 整体与拆分

| 方式 | 使用入口 | 权重/执行特点 |
|---|---|---|
| 整体 | `FlowMatchPolicy.from_checkpoint`或`from_pretrained` | 两模块在config.device执行 |
| 同进程分设备 | `FlowMatchPolicy.from_components(..., vision_device=..., action_device=...)` | 两模块都在一个Python进程，可分配不同设备 |
| 独立worker | `runtime.VisionWorker` / `runtime.ActionWorker` | 各自只加载视觉或动作PT，由调用端传递VisualPacket |

可由同一对组件导出原生同进程split目录：

```bash
python -m lerobot.policies.flowmatch export --vision-checkpoint outputs/flowmatch_100k/source_weights/components/vision.pt --action-checkpoint outputs/flowmatch_100k/source_weights/components/action.pt --execution-mode split --vision-device cpu --action-device cpu --output outputs/flowmatch_100k/split
```

**标准split目录仍有两模块的完整safetensors**，不等同动作头单独服务。真正独立进程使用两个worker文件。worker检查bundle、统计/任务metadata、episode/frame和视觉时效；默认时钟域按主机区分。跨主机需另接传输/统一时钟与调度，本目录没有RPC或真机控制循环。

## 5. 现场配置与验证

先填写`deployment.example.json`的副本，核对四路物理相机、GELLO→FR3关节顺序/符号/偏置、raw夹爪0/1对应的gPO端点以及task ID。转换时加`--deployment-config verified_mapping.json`。关节是绝对目标，不加实测q；raw夹爪会在映射时裁剪0…1。原模型夹爪存在越界输出，裁剪不代表问题已消失。

本机已用100k实权重对比旧训练环境、整体、独立组件与原生重载，合成观测/同噪声的CPU输出最大差为0。可在目标Linux电脑复跑：

```bash
python -m lerobot.policies.flowmatch verify --weights outputs/flowmatch_100k/source_weights --policy-path outputs/flowmatch_100k/whole --output outputs/flowmatch_verification_new.json
python -m pytest -q src/lerobot/policies/flowmatch/tests --confcutdir=src/lerobot/policies/flowmatch/tests
```

`assets/golden_100k.*`仅是合成参考，不含机器人图像。设置`FLOWMATCH_WEIGHTS_ROOT`可额外运行实权重测试；输出使用新目录。Windows单元测试对上游Linux专用`fcntl`仅提供“允许导入、实际锁操作必报错”的测试替身，不进入生产代码，也不表示Windows实机支持。测试没有调用数据转换、HDF5读取或机器人控制。

Linux控制电脑整链、2080/CUDA延迟、物理映射和真实任务成功率仍需现场验证。配置未填写时可离线生成raw预测，原生真机动作接口明确拒绝未确认的映射。

参考来源和DiT CC BY-NC4.0归属见[NOTICE.md](NOTICE.md)、[LICENSE-DiT.txt](LICENSE-DiT.txt)。本实现和说明有显著AI辅助，按项目AI_POLICY披露。本次按源码目录/可编辑安装交付，没有修改上游打包规则或发布wheel。
