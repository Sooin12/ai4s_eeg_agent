# 小型真实 EEG 全链路冒烟验收（2026-08-15）

## 结论先行

系统已在一个许可证明确、具有三个分日会话的真实运动想象 EEG 小样本上完成：

`Dataset-Level 理解与联网调研 → 独立 Dataset Critic → 冻结粗搜索范围 → Research Design Planner/Critic → 被试画像 → 被试级联网查文献 → 预算化完整 pipeline 搜索 → 独立 Lock Critic → one-shot frozen confirmation → Evidence Reporter → 独立 Scientific Critic`。

工程闭环完成，但科学结论严格保持 `inconclusive`：搜索 session 的 balanced accuracy 为 0.5583，冻结确认 session 为 0.5500，未达到预冻结的 0.60 成功阈值；同时只有一名被试，不能形成总体有效性结论。该结果适合证明系统会自主运行、会个体化选择、会冻结确认、会拒绝夸大，不适合宣称算法在真实 EEG 上已取得优越性能。

## 数据与许可

- 数据：BNCI Horizon 2020 数据库的 004-2014（二分类左右手运动想象）B01T 小包。
- 官方页面：<https://bnci-horizon-2020.eu/database/data-sets>
- 官方范式说明：<https://bnci-horizon-2020.eu/database/data-sets/004-2014/description.pdf>
- 许可：CC BY-ND 4.0；本地分析，不重新分发原始或标准化副本。
- 下载文件：34,203,437 bytes。
- 原始文件 SHA-256：`0da6e77ab0dab5b4aa1d2d5a6a542ac02f6768d3b7a76b0abe896ec1cf259919`。
- 验收子集：1 名被试、3 个独立会话、400 个试次（200 左手、200 右手）、C3/Cz/C4 三通道、250 Hz。
- 专家伪迹标注未被静默用于删除试次；本次 400 个试次全部保留。

重要边界：官方下载包是嵌套 MATLAB 连续记录，本次先以一次性的确定性本地步骤转换为通用 `trial × channel × sample` MAT epoch 合同，再由现有 `mat_epoch_semantic_v1` 适配器独立核验键、轴、身份、标签、有限值、平坦通道、试次数与全部源 SHA。因而本次证明的是“标准语义接入后的自主闭环”，尚不能证明任意嵌套原始容器都可零人工接入。转换逻辑没有进入项目主体，也没有添加任何数据集 ID 分支。

## Dataset-Level Agent 结果

- 正式 run：`artifacts/runs/bnci-small-dataset-level-20260815-v2`
- Adapter：`mat_epoch_semantic_v1`
- 身份冲突：0。
- 外部权威 blocker：0。
- Dataset-Level 科研周期：2；第一轮 Critic 要求修订，第二轮才通过并冻结。
- 联网来源：Crossref 与 OpenAlex；最终冻结 7 个仅作假设的 frontier directions。
- 冻结合同：`dataset_level_contract.json`，状态 `frozen_dataset_level_contract`。
- 合同合法物化后只允许 `bandpower_lda`；`csp_lda` 因合同已排除 `feature_csp` 而不会暴露给被试级 Search Agent。

Dataset-Level 阶段共报告 87,087 API tokens。该阶段的旧入口尚未使用统一付费账本，费用按同一 Kimi 价格快照从 provider usage 重算约为 0.334029 USD；原始逐调用用量保存在 `dataset_level_run.json`。

## Research Design 冻结结果

- 正式 run：`artifacts/runs/bnci-small-real-loop-20260815`
- 协议经过 2 轮 Planner/Critic 后冻结。
- 角色分配：session 01 画像；session 02 搜索与锁定；session 03 frozen confirmation。
- 主指标：balanced accuracy。
- 机会水平：0.50。
- 最低确认阈值：0.60。
- 最大允许搜索到确认下降：0.15。
- 最少不同候选：2；最多候选：5。
- frozen confirmation 最大访问次数：1；确认后禁止重开搜索。

Research Design 使用 265,649 tokens、约 0.693836 USD。为了在不改变任何科研合同的情况下完成已冻结下游，API token 总上限从 350,000 扩到 650,000；付费上限仍为 6 USD，扩展记录为 `authorities/budget_extension.json`。

## 被试画像与自主搜索

画像 session 01 的确定性测量：

- 全部值有限；无平坦或鲁棒离群通道；仅标记 1 个鲁棒离群试次，不自动排除。
- mu 候选峰 11 Hz；beta 候选峰 20 Hz。
- 8–30 Hz 单通道最大标准化效应约 0.175；beta 子带最大约 0.344（Cz）。
- 无事件前基线，Agent 明确拒绝把频段差异解释为 ERD/ERS。

Search Agent 先调用 Crossref/OpenAlex，锁定记录引用了本次实际返回的两个 DOI，然后在 session 02 用满 5 个候选预算：

| 候选 | 频带 Hz | 通道 | LDA shrinkage | 搜索 BA |
|---|---:|---|---:|---:|
| p01-wide830-all-s01 | 8–30 | all | 0.1 | 0.5250 |
| p02-beta1330-all-s01 | 13–30 | all | 0.1 | 0.4917 |
| p03-mu913-all-s01 | 9–13 | all | 0.1 | 0.5167 |
| p04-wide830-all-s05 | 8–30 | all | 0.5 | 0.5417 |
| **p05-wide830-c3cz-s05** | **8–30** | **C3/Cz** | **0.5** | **0.5583** |

这体现了“数据集级粗范围 + 被试画像级精细 pipeline”的实际差异：画像提出 11 Hz/20 Hz 和通道选择假设，Agent 对固定宽带、beta、个体化 mu、正则化强度及 C3/Cz 子集做顺序实验；最终没有机械选择个体峰窄带，而是依据当前被试的本地证据锁定 8–30 Hz 与 C3/Cz。

## 一次性确认与结论

- Lock Critic：`pass`。
- confirmation access count：恰好 1。
- 确认 session：03，160 trials。
- 确认 BA：0.5500。
- 搜索→确认变化：-0.00833，未超过 0.15 最大下降。
- 但 0.5500 未达到 0.60 成功阈值，且评估单元不足。
- 确定性决策：`inconclusive / not_all_success_thresholds_met`。
- Scientific Critic：`pass`，仅授权内部证据报告；`external_claim_authorized=false`。

下游使用 230,075 tokens、约 0.442725 USD。Dataset-Level、Research Design 与下游合计报告 582,811 tokens，按同一价格快照合计约 1.470590 USD。

## 完整性与已修复的工程缺口

- DatasetLevelContract、冻结协议、证据报告及 Scientific Critic 均已重新调用确定性 validator 通过。
- 三条审计序列分别为 125、75、94 个连续事件。
- 两条付费预算账本均成功重放 SHA-256 链。
- run manifest 中全部绑定文件的当前 SHA-256 均匹配。
- confirmation 状态为 `completed_one_shot`，没有 refit 或 reopen search。
- 原运行的 API 账本没有同步写入候选数、确定性计算秒数和确认访问数；这些量仍被原始 `pipeline_lock`、`confirmation_access.json` 和结果文件完整记录。由于原账本已关闭，未对其改写；补充核算保存在 `resource_accounting_supplement.json`。
- 通用运行时已修复：后续候选执行、研究周期、确定性计算时间和 confirmation access 会原子写入共享 `BudgetLedger`。
- Kimi 的跨 Agent 组织级 3 RPM 约束已加入共享 21 秒请求间隔，避免不同 Agent 实例分别限流。
- 可执行能力现在必须与冻结 DatasetLevelContract 求交集，防止已排除组件在下游重新出现。

## 初赛材料的准确表述

可以写：

> 系统已在一个真实、分会话、开放许可的运动想象 EEG 小样本上完成端到端自主科研冒烟测试。Agent 从数据集级认知和文献调研出发，冻结合法粗搜索范围；随后根据被试的频谱与空间特征自主检索方法、顺序运行 5 个完整 pipeline，并锁定 C3/Cz 的 8–30 Hz bandpower-LDA。锁定后仅访问一次独立会话，搜索/确认 balanced accuracy 分别为 0.558/0.550。结果未达预冻结成功阈值，系统如实给出 inconclusive 而非追认成功，证明了自主运行、个体化选择、泄漏控制与证据边界。

不要写：真实数据准确率达到先进水平、已经证明个体化优于通用方案、已经实现任意原始格式零人工接入、或该单被试结果可泛化到人群。
