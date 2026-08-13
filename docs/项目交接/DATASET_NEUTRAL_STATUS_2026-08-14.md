# 数据集无关主干状态（2026-08-14）

## 当前事实

项目主体不包含任何具体数据集名称、ID、DOI、私有目录规则、专用 adapter、专用 CLI、
权威信封、冻结协议或运行结果。真实数据只作为仓库外输入，通过标准语义合同进入。

当前 profile-ready 接入能力：

- `bids_eeg_v1`：合法 BIDS EEG；
- `mne_raw_semantic_v1`：MNE 可读容器 + 明确语义 sidecar；
- `mat_epoch_semantic_v1`：三维 MAT epoch 数组的确定性 reader；
- `declarative_semantic_v1`：尚无专用 reader 的灵活容器，明确保留内部数值核验限制。

格式扩展名只触发容器识别。进入 Dataset-Level Agent 前必须完成适配器选择、确定性
语义验证、全源 SHA 绑定和标准 `DatasetProfile` 验收。后续阶段只能消费标准合同。

## 本轮清理

已移除旧的具体数据集源码、配置、测试、清单、协议、授权、派生输出和以旧运行结果为
中心的交接/验收文档。旧运行目录已移出项目树。原始数据未删除、未覆盖。

## 通用 MAT 验收边界

`mat_epoch_semantic_v1` 从 sidecar 接收最小必要语义声明，本地程序自行生成结构和质量画像。
它支持任意被试/会话标签和可选 run 身份，不假设数字编号、固定会话数或特定目录名。
它会报告缺失 subject-session、trial-count anomalies、非有限数组和平通道，不把少 trial
自动等同于无效 run，也不执行静默排除。sidecar 可为路径中的第二身份声明提供
assertion regex；两处声明不一致时，数据集层画像会量化冲突并保留 blocker，
而被试层 loader 会 fail closed，直到外部权威证据解决身份映射。

已使用仓库外的 MAT fixture 和真实 provider 完成一次全量 Dataset-Level Agent
闭环：通用适配器被自主选中，文献检索、独立 Dataset Critic 与合同冻结均完成。
验收中发现的多身份声明冲突已转化为上述通用能力；最新合同正确保留该
blocker 并禁止下游激活。真实 fixture、sidecar、运行审计和合同仍仅保存在仓库外。

## 下一阶段

Method Engineering Agent 仍是主线，但其输入必须来自新的、数据集无关冻结合同。仓库外
验收 fixture 当前存在未解决的外部权威 blocker，因此其冻结合同不能作为下游研究执行的
授权。任何正式研究都需先解决 blocker，再用该合同的绝对路径与 SHA-256 建立一次性
`AutonomyEnvelope`。
