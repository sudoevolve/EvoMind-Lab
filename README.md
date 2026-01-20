# EvoMind Lab  
### A Time-Compressed Evolutionary Intelligence System

> We do not train intelligence.  
> We create conditions under which intelligence evolves.

---

## 🧠 项目简介

**EvoMind Lab** 是一个人工进化实验系统，目标不是训练一个更强的模型，而是构建一个**可以自我选择、自我分化、并在时间压缩条件下持续进化的智能系统架构**。

在本项目中：

- 模型参数 **保持冻结**
- 智能的变化来源于：
  - 选择机制（Selection）
  - 结构变异（Mutation）
  - 长期运行（Time）

**进化不是能力，而是“选择机制 + 时间”的结果。**

---

## 🎯 核心理念

当前主流 AI 研究主要集中在：
- 更大的模型
- 更多的数据
- 更强的算力

EvoMind Lab 关注的是另一条路径：

> **智能是否可以在没有明确目标函数的前提下，自然涌现？**

本项目尝试回答的问题包括：

- 智能是否可以不依赖梯度下降而进化？
- 行为分布是否可以在长期运行中发生迁移？
- Agent 是否会自发分化出不同“性格”和策略？
- 在时间被极度压缩的情况下，进化是否仍然成立？

---

## 🧬 系统概览

EvoMind Lab 是一个 **进化型多 Agent 系统**，其核心循环如下：

Environment ↓ Population (Multiple Agents) ↓ Evaluation (Fitness & Behavior Metrics) ↓ Selection & Reproduction ↓ Next Generation

进化不发生在模型权重中，而发生在 **Agent 的认知结构与行为策略** 中。

---

## 🤖 Agent 结构

每个 Agent 被视为一个独立的“心智个体”，包含以下组成部分：

- **Genome（基因）**
  - Prompt 结构
  - Memory 读写策略
  - 决策流程
  - 行为偏置参数（探索 / 保守等）

- **Memory（记忆）**
  - 短期 / 长期内容
  - 总结与遗忘规则

- **Behavior History**
  - 行为轨迹
  - 决策记录

- **Fitness**
  - 多目标评估得分

模型参数本身不会被训练或更新。

---

## 🧪 进化机制

### 进化单位
- Agent 的 **认知结构与行为策略**
- 而非神经网络参数

### 变异（Mutation）
- Prompt 结构扰动
- Memory 策略微调
- 行为偏置参数加噪声
- 允许偶发的大幅退化

### 选择（Selection）
采用 **多目标选择机制**，而非单一最优：

- 任务完成度
- 新颖性（Novelty）
- 行为稳定性
- 资源效率

系统关注的是 **长期可存活性**，而非一次性最优解。

---

## ⏱ 时间压缩

自然进化需要漫长时间。  
EvoMind Lab 使用算力来压缩时间：

- 多 Agent 并行运行
- 高频世代迭代
- 长期无人值守执行

在有限硬件条件下（如单张 RTX 4060），仍可模拟数千至数万代演化过程。

---

## 📁 项目结构

EvoMind-Lab/ ├── agents/            # Agent 定义 ├── evolution/         # 变异、选择、繁殖逻辑 ├── environment/       # 任务与环境定义 ├── evaluation/        # Fitness 与行为评估 ├── experiment/        # 实验入口与配置 ├── logs/              # 世代数据与谱系记录 ├── visualization/     # 可视化工具 └── README.md

---

## 🚀 当前状态

- [ ] 基础 Agent 架构
- [ ] 初代进化循环
- [ ] 多目标评估机制
- [ ] 行为演化可视化
- [ ] 长期运行实验

项目处于 **实验性研究阶段**，结果可能是失败的、不稳定的，甚至是“奇怪的”。

这正是设计的一部分。

---

## 📌 项目定位

- **Open-ended Evolutionary AI**
- **Artificial Cognitive Evolution**
- **Self-Selecting Agent Systems**
- **Beyond Gradient-Based Learning**

这不是一个商业产品，而是一个 **关于智能如何产生的工程实验**。

---

## ⚠️ 声明

EvoMind Lab 不试图构建：
- 强人工智能
- 自我意识系统
- 通用对话助手

本项目关注的是 **进化机制本身**，而非类人表现。

---

## 🧠 项目宣言

> Intelligence does not need to be designed.  
> It needs an environment where it can survive, fail, and change.

---

## 📜 License

MIT License (for experimental and research use)


---
