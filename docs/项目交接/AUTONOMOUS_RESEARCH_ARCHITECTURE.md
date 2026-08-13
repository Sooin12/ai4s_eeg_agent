# 全自动科研 Agent 闭环架构 v1.0

> 日期：2026-08-05  
> 状态：取代“每阶段人工批准”的旧编排原则；历史协议与批准产物仅作为可追溯记录保留

## 1. 评价对象

项目要验证的不是“研究者在 Agent 辅助下能否完成 EEG 分析”，而是：

> 在一次性给定研究目标、数据权限和总预算后，科研 Agent 能否以更少的顺序科研周期，为不同个体找到接近有限 oracle 的完整 BCI pipeline，并在未参与搜索的独立数据上确认，同时给出可重放证据。

用户不是流水线中的逐项审批者。用户只负责运行前的资源授权，以及许可证、伦理、不可逆数据操作和对外发布等责任事项。

## 2. 一次性自主授权

每个正式研究创建一个版本化 `AutonomyEnvelope`：

- `objective`：研究问题与允许回答的结论范围；
- `dataset_access`：允许访问的数据合同与角色候选，不包含预先泄露的确认结果；
- `resource_budget`：本地计算、API、候选执行数和最大顺序科研周期；
- `forbidden_actions`：原始数据修改、结果反向泄漏、外部发布等禁止项；
- `confirmation_policy`：锁定条件、一次性访问和确认后禁止重开搜索；
- `failure_policy`：预算耗尽、证据冲突或无法复现时允许拒绝输出强结论。

在 envelope 内不再设置人工批准门。所有科研决策由 Agent 产生，由独立 Critic 与确定性验证器共同约束，并写入不可变账本。

## 3. 六个闭环

### 3.1 数据集理解闭环

```text
Inspector → Adapter Probe → Dataset Profiler → Contract Validator
                    ↑                 ↓
              unsupported/ambiguous diagnosis
```

目标是生成证据充分的标准 `DatasetProfile`。不支持或语义歧义时输出适配器需求，而不是让 LLM 猜测数组含义。

### 3.2 研究设计闭环

```text
Protocol Planner → Protocol Critic → Deterministic Validator
       ↑                                      ↓
       └──────── structured revision ─────────┘
                              ↓ pass
                     Frozen ProtocolContract
```

Planner 自主确定数据角色、指标、统计检验、oracle、预算、停止条件和拒绝门槛。Critic 检查泄漏、统计合理性、数据契约冲突和结果前窥视。验证通过后自动冻结，不等待用户逐项批准。

### 3.3 方法发现闭环

```text
Canonical Registry + Literature Scout
                 ↓
Method Candidates → Evidence Critic → Implementability/License Tests
                 ↑                         ↓
                 └──── revise/reject ──────┘
```

本地组件库提供可执行记忆，联网发现提供认知扩展。新方向必须经过证据、许可证、接口、泄漏和最小复现检查后才能进入可执行空间。

### 3.4 被试理解闭环

```text
Deterministic Measurements → Subject Profiler → Uncertainty Check
            ↑                                      ↓
            └──────── targeted measurement ────────┘
```

确定性工具计算噪声、坏道、个体频率、ERD/ERS、可分性、稳定性和跨会话漂移。Agent 只能读取结构化摘要，并可在预算内请求补充测量，直到画像充分或明确报告不确定。

### 3.5 个体 pipeline 搜索闭环

```text
SubjectProfile + SearchSpace
             ↓
Candidate Generator → Deterministic Executor → Result Diagnoser
       ↑                                        ↓
       └──── budget-aware next experiment ──────┘
                              ↓ stop
                         Pipeline Lock
```

每次科研周期是一轮“提出完整 pipeline—执行—诊断—决定下一实验”。Agent 在固定预算内选择信息增益最大的下一轮，而不是穷举。锁定产物包含完整配置、数据边界、代码/环境、随机种子、结果和 SHA。

### 3.6 确认与解释闭环

```text
Locked Pipeline → One-shot Confirmation Controller → Confirmation Result
                                                        ↓
Evidence Reporter → Scientific Critic → final / weak / reject
```

控制器只在所有前置 hash 和账本条件满足时自动访问 frozen confirmation。确认表现下降只能进入解释和风险报告，不能反向改变 pipeline。Scientific Critic 检查结论是否由真实实验、比较、消融或反事实支持。

## 4. Agent 与确定性程序的边界

Agent 负责：语义理解、实验规划、候选选择、失败诊断、证据组织和下一步决策。

本地程序负责：原始数据读取、数值计算、训练、数据划分执行、随机种子、指标、统计检验、预算扣减、hash、阶段转换和 frozen access。

Agent 的“知识更多”不是可信度来源。可信度来自可追溯证据、独立批判、确定性执行、结果前冻结和独立确认。

## 5. 实验评价

主要比较：

- `Autonomous Guided Agent`；
- `Random Search`；
- `Global Best`；
- `Finite Individual Oracle`；
- 可选的固定专家 pipeline。

科研效率的主横轴不是聊天轮数，而是顺序科研周期数和实际候选执行数。至少报告：

- confirmation balanced accuracy、accuracy 和 kappa；
- oracle gap 随科研周期的变化；
- 达到预设 gap/性能门槛所需实验数；
- search 到 confirmation 的性能衰减；
- wall time、计算量、API token/成本和失败恢复次数；
- 个体间最终 pipeline 差异；
- 拒绝强结论的被试与负结果。

协议、门槛和 oracle 必须在对应结果可见前由研究设计闭环自动冻结。

## 6. 实现顺序

1. `AutonomyEnvelope`、统一决策记录与自动冻结协议；
2. Subject Profiler 的确定性测量工具与闭环；
3. 最小声明式 pipeline executor；
4. 预算搜索、诊断和锁定；
5. one-shot frozen confirmation 与证据报告；
6. 先合成夹具冒烟，再单被试跑通，最后进行多被试对比。

开发期间只穿插能验证当前闭环接口、泄漏防线和重放能力的必要实验，不以人工分析数据集替代 Agent 能力。
