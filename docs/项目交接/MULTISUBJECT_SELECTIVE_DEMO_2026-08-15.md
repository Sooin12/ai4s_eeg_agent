# 多被试选择性个体化 Demo 交接（2026-08-15）

> **已被 v2 交接取代。** 此文档记录 v1 在首次限额中断时的状态，仅用于历史审计。v1 后续恢复至被试 02 的 Pipeline Lock Critic，并因冻结门控与 runner 预算不一致被判 `revise`；没有访问 confirmation。正式完成状态与结果请以 `MULTISUBJECT_SELECTIVE_DEMO_V2_2026-08-15.md` 为准。

## 当前结论

多被试数据、Dataset-Level 合同、统一 pipeline incumbent 选择器、选择性个体化门控、Critic 复核和可恢复运行入口均已完成。正式四被试运行尚未完成：它安全停在 `Research Design / protocol_critic` 之前，原因是 Moonshot 账户达到组织级每日 1,500,000 token 上限。没有访问任何 subject 的 frozen confirmation session。

不得把当前状态描述成“多被试真实闭环已经跑完”或“个体化已经优于统一 pipeline”。可以描述为“多被试正式运行已建立并冻结授权边界，Dataset-Level 阶段完成，Research Planner 已产出合法提案；受外部 API 日额度限制，独立 Protocol Critic 与下游确认待续跑”。

## 数据与标准适配

- 外部测试数据：BNCI Horizon 004-2014 的 B02T、B03T、B04T、B05T。
- 原始文件目录：`data/raw/bnci-004-2014-small/`。
- 标准化目录：`data/processed/bnci-004-2014-multisubject-demo/`。
- 适配器：`mat_epoch_semantic_v1`。
- 规模：4 名被试、12 个独立会话、1,640 个 trial、C3/Cz/C4、250 Hz。
- 所有 trial 保留；专家伪迹标记只记录计数，没有静默删除。
- 角色候选仍是三个独立 session；截至本文档生成时，session 03 从未被正式或开发搜索加载。
- 嵌套原始 MAT 到标准 epoch MAT 仍是一次性本地转换步骤，因此不能宣称任意嵌套原始容器已实现零人工接入。

原始 SHA-256：

- B02T：`1c4ace3eee8d72ca184fa6995a9466939b71175b8b3a129bdaa838a85adf6473`
- B03T：`4ae1b23b4b4359151787a1a61d3acea128719677c49bfaf0113748cffece3b98`
- B04T：`e7230d6d28a9e81d3afdf7ba5d363979e0186dbbc5a73f66d4b713be020ce960`
- B05T：`53166e2cf1a97576262b3f2f27395f489af6fda15ade5c3b3cdd2c89f7df0d79`

## Dataset-Level Agent

- 正式目录：`artifacts/runs/bnci-multisubject-dataset-level-20260815/`
- 状态：`completed`。
- 冻结合同 SHA-256：`a1fbe43c5d6c0d2595df18dda1a9a30da4ebb9e2fa669fea9d4dd16203fdff9e`。
- 共 3 个科研周期。
- 第 1 轮 Critic 拒绝与三通道约束冲突的 covariance/CSP/ICA 提案。
- 第 2 轮 Critic 要求修正文献元数据证据边界、目标会话标签泄漏表述和 AutoML 的过强负面文献断言。
- 第 3 轮通过并冻结。
- 冻结可执行交集允许 `bandpower_lda`；CSP、FBCSP、协方差切空间和 MDM 等因数据合同限制不能进入下游执行。

## 新增通用能力

### 数据集统一 incumbent

`src/bci_autodiscovery/search/dataset_incumbent.py` 在标准 epoch 合同上执行同一组全通道完整 pipeline，并按被试等权计算：

`robust_score = macro_mean_balanced_accuracy - 0.25 × subject_std`

产物记录每个候选的逐被试结果、宏平均、被试间标准差、最差被试、实际执行数、计算秒数、全部配置 hash 和来源合同。验证器会重新计算排名并检查自哈希，不能把非确定性优胜者伪装成 frozen incumbent。

通用入口：`python -m bci_autodiscovery.search.dataset_incumbent_cli`。

### 选择性个体化

被试级 Search Agent 必须先执行 frozen dataset incumbent 作为强制对照。只有最佳个体候选在搜索数据上至少超过 incumbent `0.03`，才能锁定个体路线；否则强制输出 `fallback_to_dataset_incumbent`。Pipeline Lock Critic 会重新计算门控，LLM 不能通过修改理由绕过。

这使系统的主张从“强制每个人个体化”变为“Agent 根据证据决定是否个体化，并在证据不足时安全回退”。确认结果仍不能改变已经冻结的路由决定。

### 终止工具恢复

Research Design 现在会保留已被确定性工具接受的终止产物。若模型在提交合法 proposal/critique 后，仅在可选自然语言收尾请求中遇到供应商错误，编排器不会丢弃该轮产物。恢复时会从审计状态重新校验、物化并绑定 SHA，而不是人工重写。

## 正式四被试运行状态

- 运行目录：`artifacts/runs/bnci-multisubject-selective-demo-20260815/`。
- Envelope：`artifacts/authorities/bnci-multisubject-20260815/autonomy_envelope.json`。
- 完整 16 候选计划：`artifacts/authorities/bnci-multisubject-20260815/dataset_incumbent_plan.json`。
- 正式主体：02、03、04、05。
- 付费上限：6 USD；正式计划没有扩大该上限。
- Planner 首次因错误使用 `session-01/02/03` 被本地验证器拒绝，第二次提交合法提案。
- 已恢复提案：`research_design/proposal-0001.json`，SHA-256 `06af7f62e6300dd183bc72d27b675ca2a07fdcd1a5dddbc23c6d7947e8cde4d4`。
- 提案角色：01 画像、02 pipeline 搜索与锁定、03 frozen confirmation。
- 主指标：balanced accuracy；最低确认 0.60；最大搜索到确认下降 0.10；至少 4 个不同候选。
- 当前状态：`failed_recoverable / protocol_critic`。
- 阻塞原因：Moonshot 组织每日 token 限额；不是代码预算超限，也不是协议或数据失败。
- confirmation access count：全部被试均为 0。

恢复入口已经加入 `scripts/run_standard_epoch_multisubject_demo.py --resume`。账户日额度恢复后，以原参数追加 `--resume` 即可；脚本会使用 `research_design/budget_extension.json` 中仍位于原 Envelope 内的续跑分配，不会重做 Planner 或修改提案。

## 开发探针（不得冒充正式结果）

在正式候选计划冻结前，曾只用 session 02 对 8 个 bandpower 候选做非正式开发探针。最稳健者是 `4–40 Hz + all channels + shrinkage 0.1`，四被试宏平均搜索 BA 约 0.585，但该结果不能作为正式 incumbent 结论。

为防止开发探针影响正式计划，冻结计划包含注册表中全部 4 个固定频带 × 4 个 shrinkage，共 16 个候选；未根据探针增加或删除候选。最终 incumbent 必须由正式运行重新执行并冻结。

## 工程验收

- 新增和相关定向测试通过。
- 全量 113 项 pytest 子进程退出码为 0。
- `compileall` 通过。
- `git diff --check` 通过。
- 数据集专用名称没有进入 Agent 编排、门控或选择器；具体名称只存在于外部数据、运行产物和本交接记录。
- 用户正在编辑的初赛 Word 文档保持未暂存、未提交。
