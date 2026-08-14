# 全线路个体化 BCI Agent Demo 交接

## 结论

2026-08-15 已在三名异质确定性合成被试上跑通真实模型驱动的下游科研闭环。该结果证明工程编排、
个体化搜索、阶段隔离、一次性确认和证据门可以协同运行；不证明任何方法在真实 EEG 上优于基线。

运行目录：`artifacts/runs/individualized-demo-20260815-v3/`

面向展示的两个入口：

- `DEMO_PRESENTATION.md`：中文汇总表和解释边界；
- `demo_presentation_bundle.json`：绑定逐被试锁、确认结果、最终报告、冻结协议、预算账本和 SHA-256。

## 实际闭环

```text
DatasetLevelContract
  → Research Design Planner
  → independent Protocol Critic / revision / freeze
  → Subject Profiler（确定性画像，原始数组不进入模型）
  → Crossref + OpenAlex subject-method search
  → budgeted complete-pipeline experiments
  → independent Pipeline Lock Critic
  → one-shot frozen confirmation
  → Evidence Reporter
  → independent Scientific Critic
  → deterministic three-subject aggregate gate
```

Research Design 经一次自主修订后在第 2 个周期冻结。三名被试均独立完成画像、文献、搜索、锁定、
确认与报告；确认访问记录均为 1，确认后没有选择、重拟合或重开搜索。

## 个体化结果

| 被试 | 隐藏表型（仅工程验收使用） | Agent 锁定 pipeline | 搜索候选 | 搜索 BA | 确认 BA |
|---|---|---|---:|---:|---:|
| subject-mu | 10 Hz 功率型 | 8–30 Hz、全通道、CSP-6、shrinkage LDA | 2 | 1.000 | 1.000 |
| subject-csp | 20 Hz 协方差型 | 8–30 Hz、全通道、CSP-4、shrinkage LDA | 3 | 1.000 | 1.000 |
| subject-beta | 17 Hz 个体 beta 功率型 | 13–30 Hz、P3/P4、bandpower、shrinkage LDA | 3 | 1.000 | 1.000 |

三条锁定 pipeline 的完整 SHA-256 均不同，覆盖两种特征族。第三名被试体现了画像到精细搜索的关键
变化：Agent 从全通道 CSP 路线切换为命名通道 `P3/P4` 的 beta bandpower 路线。每名被试的锁定均
引用 3 个联网返回的稳定论文 ID；这些题录/摘要仅支持“为何值得实验”，最终选择仍由本地交叉验证决定。

## 为什么逐被试是 inconclusive，而总体是 success

冻结协议的 aggregation unit 是 `subject`，并预先要求至少 3 个可评估 subject。单个被试即使搜索和
确认 BA 均为 1.000，也不能冒充总体证据，因此逐被试 Evidence Report 正确保持 `inconclusive`。

三名被试全部完成后，独立的确定性聚合器才检查：

- 三次确认是否均为 one-shot；
- 每名被试是否满足最少独立候选数；
- 宏平均确认分数和搜索到确认下降是否越过冻结阈值；
- Pipeline Lock Critic 与 Scientific Critic 是否全部通过；
- 可评估 subject 数是否达到 3。

上述门全部通过后，总体工程结论为 `success`。该 success 的作用域固定为
`synthetic_engineering_demo_aggregate`，`external_scientific_claim_authorized=false`。

## 预算与恢复

初始 500,000 token 上限在第三名被试画像阶段 fail-closed。用户明确授权把总 token 上限提高到
750,000，费用上限仍保持 12 美元。系统没有改写原 `AutonomyEnvelope` 或原账本，而是新增：

- `budget_extension.json`：记录原 envelope/ledger SHA、授权差异和禁止事项；
- `budget_extension_ledger.jsonl`：只管理新增 250,000 tokens，继续使用 hash chain。

合计使用 659,571 tokens，实际计费约 1.649 美元。预算补充没有改变研究目标、数据范围、冻结协议、
阈值或确认政策，也没有重跑前两个被试。

运行中发现并修复了三个通用合同问题：

1. v2 `decision_policy` 比旧判定器多 `policy_version` 与 `minimum_evaluable_units`；判定器现能安全解析；
2. Pipeline Search 应读取正式 `stopping_policy`，不能依赖旧 `stopping_conditions`；
3. Evidence Reporter 的合法引用语法是 `artifact#field`，现已写进模型提示与工具说明。

失败、修复和恢复均保留在同一 append-only `audit.jsonl` 中。

## 验证状态

- 新增个体峰频候选、画像排名通道集、通道名绑定确认、subject-method 文献工具与证据锁；
- 新增统一 Demo CLI、预算补充恢复、总体展示 bundle；
- 108 项单元测试全部通过；
- 运行产物在 `artifacts/runs/`，不进入 Git；源码和交接文档保持数据集无关。
