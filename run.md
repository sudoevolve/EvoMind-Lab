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

- 云端商用模型接入与统一抽象（当前主要支持本地 Ollama；也保留规则/统计策略）
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

### 2.3 用 projects 配置跑（推荐）

用 `--project` 直接加载 `projects/<name>/project.json`：

```bash
py -m experiment.run --project crypto_backtest --generations 10 --population 24 --survivors 12 --elite 3 --mutation 0.25 --out logs/crypto_backtest
```

`csv_market` 的数据文件支持两种格式：

- 带表头的 OHLCV（至少包含 `close` 列）
- Binance Kline 导出（无表头，逗号分隔，第 5 列为 close）

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

### 4.2 一键生成“可读分析报告”

把某次 run 目录（或 generations.jsonl 文件）丢给分析器：

```bash
py -m experiment.analyze --run logs\crypto_backtest\20260302_090929 --top 5 --per-gen
```

分析器会做两件事：

- 把报告打印到终端
- 同时把报告写入 run 目录的 `analysis.txt`（方便后续查看与分享）

### 4.3 常见疑问（FAQ）

#### Q1：终端一关分析结果就没了，怎么办？

A：用分析器时会自动把报告写到 run 目录的 `analysis.txt`。例如：

```bash
py -m experiment.analyze --run logs\crypto_1m\20260302_105610 --top 5 --per-gen
```

运行后你会在该目录看到：

- `analysis.txt`

#### Q2：我改了 `logs/.../config.json`，为什么再次运行没生效？

A：`logs/<run>/config.json` 是“那次运行的参数快照”，只用于留档，不会被 `experiment.run` 读取来重新跑。

要让参数生效，你应该：

- 用命令行参数运行（例如 `--generations/--population/--horizon/...`），或者
- 修改 `projects/<name>/project.json` 里的 `experiment` 配置，再用 `--project <name>` 跑新的一次

#### Q3：我保留的好个体在哪里？“精英保留”到底保留了什么？

A：保留的是“Genome（基因）”，不是同一个 `agent_id`。

- 每一代所有个体都在 `generations.jsonl` 的 `agents[]` 里（包含 `agent_id / genome / objectives / episode`）
- 精英保留会把排名靠前的父代 genome 直接复制到下一代，但子代会分配新的 `agent_id`
- 父子关系在每一代记录的 `lineage_next` 里：`下一代 child_id -> (父代 parent_id,)`

实际操作上，拿“好个体”做复现/对比通常有两种方式：

- 方式 1：看 `analysis.txt` 的 Top N，拿其中的 `agent_id` 去 `generations.jsonl` 里搜索，复制它的 `genome`
- 方式 2：沿 `lineage_next` 追溯某个 `agent_id` 的祖先，观察哪些基因在多代里被保留下来

#### Q4：我怎么找到“最全的 K 线数据”？

A：做 1m 回测最常用、覆盖广且可复现的数据源是交易所官方历史数据。

- Binance 官方历史数据（推荐）：data.binance.vision  
  - 常见为按 `symbol / interval / year / month` 打包下载的 CSV/ZIP
  - 本仓库的 `csv_market` 支持：带表头 close 或 Binance kline 无表头格式（第 5 列 close）

更“全”的（tick/逐笔、多交易所）通常是付费数据；如果你只是做 1m 策略筛选，官方历史数据一般足够。

#### Q5：为什么我没感受到“进化”？AI 会一次次变好吗？

A：这里的“变好”来自进化算法（选择/繁殖/变异）在搜索 Genome 空间，而不是训练大模型权重。

它通常呈现为：

- “最好个体”更可能变好，但不保证每一代都更好（行情噪声 + 随机变异会让曲线抖动）
- 多目标选择会保留“新颖/稳定/高效率”的个体，因此利润不一定单调上升

如果你几乎看不到进化效果，最常见原因是“信号不够强/评估不够稳”：

- horizon 太短（1m 数据用几十/几百步很容易全是噪声）
- 种群/代数太小（搜索强度不够）
- 评估只跑 CSV 开头一段（容易对开头行情过拟合，换区间就失效）

经验上更容易看到趋势的做法：

- 把 horizon 拉长到几千~上万步，并增加 generations/population
- 用多段窗口/随机起点（或 walk-forward：训练段选、测试段验）来做更可信的筛选

#### Q6：我想用大模型（Ollama），为什么 GPU 没动？怎么确认真的在走模型？

A：GPU 没动通常只有两种原因：要么你没启动 Ollama 服务，要么本次运行没用 `ollama` 策略（还在用 `rule/ucb1/...` 这种 CPU 规则/统计策略）。

1) 启动 Ollama 服务（Windows 常见路径）：

```bash
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" serve
```

2) 确认服务与模型都可用：

```bash
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" list
```

3) 让本仓库强制使用 Ollama（推荐用环境变量，方便随时开关）：

```bash
$env:EVOMIND_FORCE_STRATEGY='ollama'
$env:EVOMIND_OLLAMA_MODEL='qwen3:8b'
$env:EVOMIND_OLLAMA_URL='http://127.0.0.1:11434'
$env:EVOMIND_OLLAMA_TIMEOUT='60'
py -m experiment.run --project crypto_backtest --generations 1 --population 2 --survivors 1 --elite 1 --horizon 16 --out logs/crypto_llm
```

4) 如何确认“真的走了模型”：

- 观察 Ollama 控制台窗口会持续打印请求/加载信息
- `analysis.txt`/`generations.jsonl` 里的“策略”字段显示的是 genome 的 `decision_strategy`，如果你用的是 `EVOMIND_FORCE_STRATEGY` 强制策略，字段可能仍显示为 `rule/ucb1/...`，但实际动作来源是 Ollama

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

1) 选一个模型来源（本仓库当前支持本地 Ollama）
2) 把模型输出约束到 action_space（只允许输出一个动作字符串）
3) 对不合法输出做回退（随机合法动作）

### 6.1 推荐的最小接入方式：做一个 LLM 驱动的 ActionPolicy

当前已经提供了最小的 Ollama 接入：把 `decision_strategy` 设为 `ollama`，即可通过 `http://localhost:11434` 走 `/api/chat` 选动作（只输出动作字符串）。

常用环境变量：

- `EVOMIND_FORCE_STRATEGY=ollama`：强制所有 agent 使用 Ollama
- `EVOMIND_OLLAMA_MODEL=qwen3:8b`：指定模型名
- `EVOMIND_OLLAMA_URL=http://localhost:11434`：指定服务地址
- `EVOMIND_OLLAMA_TIMEOUT=10`：请求超时秒数

如果你不想用模型，也可以继续用 `rule / ucb1 / softmax / epsilon_greedy / random`。

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

