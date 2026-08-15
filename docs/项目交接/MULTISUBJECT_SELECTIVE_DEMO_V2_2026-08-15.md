# 多被试选择性个性化 Demo v2 交接（2026-08-15）

## 最终状态

正式运行已完成：

- 运行目录：`artifacts/runs/bnci-three-subject-selective-demo-20260815-v2/`
- 状态：`completed`
- 被试：03、04、05
- 每名被试：独立画像、联网文献发现、8 个完整 pipeline 候选、Pipeline Lock Critic、一次 frozen confirmation、证据报告与 Scientific Critic
- 三名被试的 Lock Critic 和 Scientific Critic 均为 `pass`
- confirmation access 均为 1，且均在 pipeline lock 后消费
- 对外科研结论仍未授权；产物只属于内部 Demo 证据

## 为什么 v1 不进入正式结论

保留但不采用的运行：`artifacts/runs/bnci-multisubject-selective-demo-20260815/`。

v1 在被试 02 的 Lock Critic 阶段被正确拦截。冻结协议要求至少 8 个个体候选，但原 runner 仅为每名被试分配 5 个科研周期；同时原重试分账的预留总数超过了 Envelope 的全局上限。被试 02 的个体搜索结果已经可见，因此没有事后放宽协议或改写锁定，而是：

1. 保留 v1 全部失败审计；
2. 将被试 02 明确排除出 v2 正式证据；
3. 在尚未执行个体搜索的 03–05 上建立新的 `AutonomyEnvelope`；
4. 在任何 v2 个体结果可见前重新冻结协议和预算。

该处理没有访问 v1 的任何 frozen confirmation 数据。

## v2 预算闭合

三被试配置使预注册要求与总 Envelope 同时可满足：

- dataset incumbent：16 个候选、48 次被试级执行、16 个科研周期；
- 个体搜索：每人 8 个候选，共 24 次执行、24 个科研周期；
- 合计：72/100 candidate executions，40/40 research cycles；
- API token 上限 2,000,000，费用上限 6 USD；
- 重试预留 22/24，恢复预留 4/8；
- frozen confirmation：每名被试最多一次。

runner 已改为根据 dataset incumbent 网格消耗和全局 Envelope，自动计算所有被试可获得的最大等额候选预算；不再硬编码每人 5 个候选。重试预算也按全局上限分配。

## dataset-wide incumbent

冻结候选为完整的 16 配置固定网格。最终 incumbent：

- pipeline：`dataset-wide-bp4-40-s0`
- family：`bandpower_lda`
- 频带：4–40 Hz
- 通道：全部通道
- LDA shrinkage：0
- 三被试宏平均搜索 balanced accuracy：0.61865
- subject std：0.01930
- robust score：0.61383
- worst-subject score：0.59167

## 选择性个性化结果

| 被试 | 路线 | 搜索 BA | 确认 BA | 确认结论 |
|---|---|---:|---:|---|
| 03 | fallback to dataset incumbent | 0.59167 | 0.48125 | refuse |
| 04 | fallback to dataset incumbent | 0.63571 | 0.70625 | inconclusive |
| 05 | personalized，13–30 Hz、全部通道 | 0.66429 | 0.57500 | inconclusive |

路线计数：2 名回退统一 incumbent，1 名进入个性化路线。

结果不能证明个性化 pipeline 优于统一 pipeline。它证明的是更基础、也更适合作为当前 Demo 主张的能力：系统能够自主画像、查找方法证据、运行完整候选、由独立 Critic 审核锁定、选择性回退，并在独立确认不支持强结论时拒绝夸大。

## 可用于初赛材料的准确表述

可以表述：

> 我们已经在三名真实 EEG 被试上完成端到端自主闭环。系统不是强制为每名被试更换 pipeline，而是先冻结数据集级稳健 incumbent，再依据个体搜索证据选择性个性化；本次 Demo 中两名被试自动回退，一名进入个性化路线。所有路线均经独立 Critic 审核和一次性独立会话确认，系统对未达到预注册证据门槛的结果输出拒绝或不确定结论。

不得表述：

- 个性化方法已经显著优于统一 pipeline；
- 三被试结果证明普遍有效；
- 该 Demo 已形成可对外发表的科学结论。

## 关键产物

- 总汇总：`artifacts/runs/bnci-three-subject-selective-demo-20260815-v2/multi_subject_summary.json`
- 运行清单：`artifacts/runs/bnci-three-subject-selective-demo-20260815-v2/run_manifest.json`
- 冻结协议：`artifacts/runs/bnci-three-subject-selective-demo-20260815-v2/research_design/frozen_protocol.json`
- dataset incumbent：`artifacts/runs/bnci-three-subject-selective-demo-20260815-v2/dataset_incumbent/dataset_pipeline_incumbent.json`
- 被试证据：`artifacts/runs/bnci-three-subject-selective-demo-20260815-v2/subjects/{03,04,05}/`
- 审计链：`artifacts/runs/bnci-three-subject-selective-demo-20260815-v2/audit.jsonl`
