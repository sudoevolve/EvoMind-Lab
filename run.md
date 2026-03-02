# EvoMind-Lab 运行指南（run.md）

本仓库实现了一个“冻结模型权重、进化发生在认知结构/策略层”的最小可运行原型：在一个时间压缩的环境里，多代运行 → 多目标评估 → 多目标选择 → 繁殖/变异 → 产生日志。

## 0. 先说结论：都完善了吗？

没有。README 里描述的是愿景与路线图；当前代码是“能跑起来、能产生日志、能迭代”的最小原型（MVP）。

你现在已经能做的：

- 跑多代进化（选择/繁殖/变异/谱系记录）
- 记录每代所有 agent 的 genome、目标分数、动作序列
- 多目标选择（task/novelty/stability/efficiency）
- 在日志里生成简单的 ASCII 趋势条（sparkline）

你现在还没有的（需要后续做）：

- 真实大模型接入（目前 Agent.act 还是规则/统计策略，不会调用 LLM）
- 并行、多环境任务集、长期无人值守的鲁棒运行（目前是单进程串行基线）
- 更丰富可视化（目前只有日志与 ASCII sparkline）

---

## 1. 环境要求

- Python：>= 3.10（本仓库已在 Windows + Python 3.13 验证）
- 不需要额外依赖（当前版本只用标准库）

建议用 `py`（Windows Python Launcher）来运行。

---

## 2. 快速开始：运行一次进化实验

### 2.0 一分钟跑起来（小白版）

1) 打开 PowerShell
2) 进入项目根目录（把路径换成你自己的）：

```bash
cd C:\Users\Administrator\Documents\GitHub\EvoMind-Lab
```

3) 跑一个很小的例子（先确认能跑通）：

```bash
py -m experiment.run --generations 3 --population 8 --survivors 4 --elite 1 --mutation 0.25 --out logs/run_demo
```

4) 你会看到它打印出一个路径，例如：

```
logs\run_demo\20260302_090929
```

5) 打开这个目录，看生成的两个文件：

- `config.json`
- `generations.jsonl`

### 2.1 在项目根目录运行（推荐）

在仓库根目录 `EvoMind-Lab/` 下执行：

```bash
py -m experiment.run --generations 30 --population 24 --survivors 12 --elite 3 --mutation 0.25 --out logs/run
```

运行完成会输出一个目录路径，例如：

```
logs\run\20260302_090929
```

### 2.2 在 experiment 子目录运行

如果你当前目录是 `EvoMind-Lab/experiment/`，使用：

```bash
py -m run --generations 30 --population 24 --survivors 12 --elite 3 --mutation 0.25 --out ..\logs\run
```

> 说明：从子目录运行时，`py -m experiment.run` 会因为模块搜索路径问题找不到 `experiment` 包；因此提供了 `py -m run` 的方式。

---

## 3. 参数说明

- `--seed`：随机种子（默认 0）
- `--generations`：世代数
- `--population`：种群规模
- `--survivors`：每代保留的候选父代数量（多目标选择后）
- `--elite`：精英保留数量（直接复制进入下一代，但会分配新 ID）
- `--mutation`：变异率（对 Genome 的各字段做随机扰动）
- `--arms`：Bandit 臂数（默认 8）
- `--horizon`：每个 episode 的最大步数（默认 32）
- `--no-stop`：关闭 stop 动作（默认允许 agent 选择 stop 提前结束）
- `--out`：输出目录（默认 `logs/run`）

---

## 4. 输出文件与数据格式

每次运行会创建一个 run 目录：

- `config.json`：本次运行的参数快照
- `generations.jsonl`：逐代日志（JSON Lines，每行一个 generation 记录）

`generations.jsonl` 每行大致结构：

- `generation`：代号
- `mean.task / mean.novelty`：当前代均值（示例只记录了 task + novelty 的均值，其他目标在每个 agent 的 objectives 中）
- `spark.task / spark.novelty`：最近若干代均值的 ASCII sparkline
- `agents[]`：每个 agent 的：
  - `agent_id`
  - `genome`：可变“认知结构/策略”配置（冻结权重理念下，这里相当于可进化参数）
  - `objectives`：四目标得分（task/novelty/stability/efficiency）
  - `episode`：本次 episode 的 action 序列与若干统计
- `lineage_next`：下一代子代 id → 父代 id（用于追踪谱系）

### 4.1 小白怎么看结果（最实用的 3 个动作）

下面这些命令都在“项目根目录”执行（先 `cd EvoMind-Lab`）。

1) 看最新一次运行输出在哪：

```bash
dir logs\run_demo
```

2) 打印某一代里 task 最高的 agent（把路径换成你自己的 run 目录）：

```bash
py -c "import json; p=r'logs\\run_demo\\20260302_090929\\generations.jsonl'; best=None; \
    \nfor line in open(p,'r',encoding='utf-8'): \
    \n  g=json.loads(line); \
    \n  top=max(g['agents'], key=lambda a: a['objectives']['task']); \
    \n  best=(g['generation'], top['agent_id'], top['objectives']['task']); \
    \n  print('gen',best[0],'best',best[1],'task',round(best[2],4))"
```

3) 把每代的 mean.task 打印成一列（方便你看趋势）：

```bash
py -c "import json; p=r'logs\\run_demo\\20260302_090929\\generations.jsonl'; \
    \nfor line in open(p,'r',encoding='utf-8'): \
    \n  g=json.loads(line); print(g['generation'], round(g['mean']['task'],4), g['spark']['task'])"
```

---

## 5. 代码入口（你要改什么看这里）

### 5.1 主循环入口

- 世代循环与落盘：`experiment/core.py` 的 `run()`  
  - 包含：运行 episode → 评估 → 选择 → 繁殖/变异 → 写日志

### 5.2 Agent 决策（当前是“无模型”的基线）

当前 `Agent.act()` 是一个 bandit 风格的策略选择器（epsilon-greedy / softmax / ucb1 / random），不调用真实 LLM：

- `agents/agent.py` 的 `act()` 与 `_act_*` 系列方法

### 5.3 Genome（可进化的“认知结构/策略”载体）

- `agents/genome.py`：存储 prompt 模板、记忆写入策略、决策策略、若干 bias 与容量上限
- `Genome.mutate()`：在变异率控制下随机扰动这些字段

---

## 6. 如何接入真实模型（LLM / VLM / 本地推理）

本仓库的核心思想是：**模型权重不变，进化发生在“如何使用模型”的层面**。落地到工程上，一般会把下面这些东西当作 Genome 的基因：

- 系统提示词模板（prompt_template）
- 工具使用与约束（tool policy）
- 记忆写入/压缩/检索策略（memory strategy）
- 规划/反思/自检流程（chain: plan → act → reflect）
- 采样参数（temperature/top_p/max_tokens/stop 等）

### 6.0 小白版：接入真实模型要做什么？

一句话：把现在 `Agent.act()` 里“规则选择动作”的逻辑，替换成“调用模型选动作”，并且**只能从 action_space 里选**。

你需要做 3 件事：

1) 选一个模型来源（OpenAI / Azure / 本地 Ollama / vLLM / llama.cpp 等）
2) 写一个“模型调用 client”（把 prompt 发出去，拿回文本）
3) 写一个“动作选择 policy”（把 observation + action_space 变成 prompt，调用 client，解析为合法动作）

目前仓库里还没有这些文件，所以这一节给的是“可照着做的接入方式”，不是已经默认完成的功能。

### 6.1 推荐的最小接入方式：做一个 LLM 驱动的 ActionPolicy

建议把“动作选择”从 `Agent` 中抽出来，做成可插拔策略层。例如：

1) 新增一个模块（示例命名）：

- `models/llm_client.py`：封装模型调用（OpenAI / Azure / 本地 vLLM / Ollama / llama.cpp 等）
- `agents/policy.py`：定义一个 `Policy` 接口：`choose_action(observation, agent)->str`
- `agents/policies/llm_policy.py`：实现用 LLM 产出动作

2) 在 `Agent.create()` 或 `Agent` 构造时注入 policy（或者在 `act()` 内根据 genome 决定用哪个 policy）。

**动作空间约束非常关键**：环境会在 `observation["action_space"]` 给出允许动作列表，你需要让模型只输出其中之一。

一个实用的 prompt 结构（伪代码，仅示意）：

```
SYSTEM: {genome.prompt_template}
USER:
你处在 bandit 环境，允许动作：{action_space}
当前观察：{observation}
从允许动作中选择一个，直接输出动作字符串，不要输出其它内容。
```

然后将模型输出做一次强制对齐：

- 如果输出不在 action_space：回退到随机合法动作
- 可加入简单的 parse（比如取第一行、strip 引号）

### 6.2 接入 OpenAI / 兼容 API 的注意点（安全与工程）

- 不要把 API Key 写入代码或日志；使用环境变量读取
- 日志里只记录必要信息（action、reward、目标分数），不要直接落盘完整 prompt（除非你明确想做审计）
- 给模型输出加“安全护栏”：严格白名单动作、最大 token、超时与重试

### 6.3 如何让“进化”真正影响模型行为

接入模型后，你需要把 Genome 的字段真正用于推理调用：

- `prompt_template`：拼接到 system prompt
- `memory_write_strategy`：控制 episode 内如何写 memory、何时 consolidate、如何压缩
- `decision_strategy`：可以不再是 bandit 的 epsilon/ucb，而是“LLM 计划 + 规则选择”或“LLM 直接选择”
- `biases`：映射到 temperature / top_p / exploration 强度 或者工具使用倾向

这样，选择与繁殖时才是在“认知结构空间”里做搜索，而不是仅在随机噪声里波动。

---

## 7. 常见问题（FAQ）

### 7.1 报错：No module named 'experiment'

原因：你在 `EvoMind-Lab/experiment/` 子目录里运行了 `py -m experiment.run`。

解决：

- 回到根目录运行 `py -m experiment.run ...`
- 或在 experiment 目录运行 `py -m run ...`

### 7.2 如何验证代码可运行

```bash
py -m compileall .
py -m unittest discover -s tests -p "test*.py" -q
```

