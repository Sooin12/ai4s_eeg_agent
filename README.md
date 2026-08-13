# 个体化 BCI 流水线自主发现系统

本仓库用于开发、验证并展示一个面向个体化运动想象 BCI 的可审计科研探索系统。

当前系统采用一次性自主授权模式：用户在运行开始前给定研究目标、数据权限和总预算，随后由多层 Agent 闭环自主完成协议设计、被试画像、方法发现、预算搜索、pipeline 锁定、独立确认和证据报告。中间科研决策不逐项请求人工批准；确定性验证器、独立 Critic、阶段账本和一次性 frozen access 负责防止泄漏与越权。详见 `docs/项目交接/AUTONOMOUS_RESEARCH_ARCHITECTURE.md`。

## 目录约定

```text
ai4s/
├─ configs/                 # 数据、实验协议和搜索空间的版本化配置
├─ data/                    # 本地数据；大文件默认不进入 Git
│  ├─ raw/                  # 下载后的原始文件，只读保存
│  ├─ interim/              # 解压、转换或清洗后的中间数据
│  ├─ processed/            # 可直接进入实验的标准化数据
│  └─ manifests/            # 文件清单、哈希、数据验收结果
├─ docs/
│  ├─ 项目交接/             # 项目章程、架构和当前上下文交接
│  ├─ 相近研究文献/         # 论文与比赛手册
│  └─ 初赛提交/             # 模板、提交稿和提交检查材料
├─ experiments/
│  └─ specs/                # 可复现实验定义；不存放运行结果
├─ src/bci_autodiscovery/   # 可复用的正式源码
│  ├─ data/                 # 数据发现、读取、校验和切分
│  ├─ profiling/            # 被试画像与数据特征
│  ├─ pipelines/            # 预处理、特征、模型流水线
│  ├─ search/               # 预算约束搜索与选择策略
│  ├─ evaluation/           # 指标、跨会话验证和确认门
│  └─ reporting/            # 证据表、图和审计报告
├─ tests/
│  ├─ unit/                 # 不依赖真实大数据的快速测试
│  ├─ integration/          # 真实文件或完整链路测试
│  └─ fixtures/             # 小型、可提交的测试样例
├─ scripts/                 # 下载检查、数据验收等命令行入口
├─ artifacts/runs/          # 每次运行的日志、模型和中间产物
├─ agent_outputs/           # Agent 面向用户发布的分阶段、可追溯派生视图
└─ tmp/                     # 随时可以丢弃的临时文件
```

## 放置规则

- 下载的数据首先放入 `data/raw/`，不要手工改写。
- 从原始数据生成的可重建文件放入 `data/interim/` 或 `data/processed/`。
- 数据文件清单、SHA-256、形状和标签检查结果放入 `data/manifests/`。
- 每个实验的输入配置放入 `experiments/specs/`；运行输出进入 `artifacts/runs/<run_id>/`。
- Agent 的用户可读输出按数据集发布到 `agent_outputs/<dataset_id>/`，索引中绑定原始路径和 SHA-256。
- 临时探索代码不得长期留在根目录；稳定后进入 `src/`，一次性维护命令进入 `scripts/`。

## 当前入口

- 最新开发交接：`docs/项目交接/DATASET_NEUTRAL_STATUS_2026-08-14.md`
- 数据语义合同 schema：`configs/dataset_semantics.schema.v1.json`
- 主流 EEG 格式与适配政策：`configs/dataset_adapter_catalog.v1.json`
- Agent 用户输出说明：`agent_outputs/README.md`
- 全自动科研闭环架构：`docs/项目交接/AUTONOMOUS_RESEARCH_ARCHITECTURE.md`

## Dataset-Level Agent

第一个完整Agent是数据集层Agent。它由职责受限的Dataset Profiler、Literature Scout和
独立Dataset Critic组成，并用确定性程序完成粗空间构建、证据账本复算与合同冻结：

```text
Inspect/Adapter → DatasetProfile → Canonical Coarse Space
                                     ↓
                         Crossref + OpenAlex discovery
                                     ↓
                         Dataset Critic → freeze/revise
                                     ↓ pass
                          DatasetLevelContract
```

从原始数据运行统一入口时，所选真实模型供应商会驱动三个子Agent；Profiler中的原始EEG读取、
结构统计和画像校验仍只在本地确定性工具内执行。Mock Profiler仅用于显式离线回归，不参与正式
验收运行。

统一入口会运行整个闭环。文献归纳和Critic会产生付费模型调用，因此必须显式选择供应商：

```powershell
python -m bci_autodiscovery.agents.dataset_level_cli `
  --dataset-profile artifacts/runs/<profile-run>/dataset_profile.json `
  --component-registry configs/component_registry.v0.json `
  --provider kimi
```

也可以改用`--dataset-root`与`--validation`从结构探测开始。完整产物包括标准
`DatasetProfile`、非可执行粗搜索空间、逐查询逐来源SQLite证据账本、候选前沿方向、
独立Critic结论、冻结`DatasetLevelContract`与不可追加覆盖的JSONL审计轨迹。原始EEG
不进入大模型上下文。当前具有通用 BIDS-EEG、MNE raw semantic、MAT epoch semantic
和 declarative semantic 画像适配器；此外会识别 EDF/BDF、BrainVision、EEGLAB、FIF、GDF、CNT、MFF、
XDF、NWB、MAT/HDF5/NumPy 等格式。格式识别不等于语义验收：缺少适配器或映射
sidecar 时会停止，不猜测数组轴、事件含义或数据角色。

非BIDS新数据可在数据根目录提供受约束的`dataset_semantics.json`。MNE可读容器会
在本地核对头信息和annotations；通用MAT epoch reader会核验对象、轴、身份、标签、
有限值、平通道和trial计数。sidecar可声明第二身份断言；两处身份不一致时会
量化为外部权威blocker，并禁止被试级加载。其他灵活容器会强制声明对象/流、数组轴、事件映射和
数据角色。所有路径均对全部源文件绑定SHA-256。详见
`docs/项目交接/FORMAT_ADAPTER_ACCEPTANCE_2026-08-12.md`。

### 粗搜索范围的职责边界

粗搜索范围只依赖标准`DatasetProfile`与版本化组件库，不依赖研究协议。只有数据、物理、
模态、范式或采集结构的硬冲突可以排除组件；会话角色、评价指标、实验预算、被试选择和
确认访问均留给后续Agent。独立运行确定性构建器的命令为：

```powershell
python -m bci_autodiscovery.agents.search_cli `
  --dataset-profile artifacts/runs/<profile-run>/dataset_profile.json `
  --component-registry configs/component_registry.v0.json `
  --provider mock
```

成熟度、成本与实现状态只是下游调度注释，不会缩小本层认知范围；本层不会训练模型或
激活方法。

### 联网判断的证据边界

Literature Scout会对画像生成的每一个`query_id/source_name`组合分别访问Crossref与
OpenAlex，并将超出本地组件库的方向记录为带论文ID、适用条件、限制、画像字段绑定和
未来协议要求的`dataset_frontier_hypothesis`。摘要/题录只能支持“值得后续实验”的假设，
不能证明效果。Dataset Critic通过后冻结的仍然是认知合同，不是可执行方法清单；实现、
许可证、接口、泄漏测试和最小复现实验属于后续方法工程/研究设计Agent。

候选方向必须从运行上下文给出的精确`DatasetProfile`叶字段目录复制画像绑定路径；工具schema、
本地处理器和独立Dataset Critic会三重校验。模型在检索或记录失败后输出最终文本不会被视为完成，
确定性完成门会返回缺口并要求Agent在预算内自主修复。正式运行拒绝旧
`constraints.requires_human_decision`画像，只接受已区分科研设计决策和外部权限阻塞的当前合同。
Critic finding同时绑定责任层；当前文献修订环只接受`owner=literature_scout`的阻塞修订，避免把
画像适配器或组件库问题交给无权修改它们的Agent。修订周期复用已审计的检索证据，不会为了修改
一条文献归纳而重复访问全部Crossref/OpenAlex查询。
完整证据覆盖的修订不会再向模型暴露搜索工具；确定性工具状态完成后也不要求额外“最终说明”模型
回合。统一CLI记录进程PID与终态，恢复入口会拒绝仍在运行或已经完成的源run，防止外层终端超时后
误启重复付费任务。

## Research Design 合同边界

Research Design 的正式入口不再接受裸 `DatasetProfile`。`AutonomyEnvelope` schema v2必须精确
授权一个冻结 `DatasetLevelContract` 的绝对路径和SHA-256；确定性loader会复核合同状态、
Dataset Critic pass、source draft、全部provenance SHA、当前约束语义和未越权的
`stage_boundary`。Planner、独立Critic、修订器、协议Loop和冻结器均从该合同派生画像上下文。

冻结协议会再次记录同一个 `DatasetLevelContract`、`AutonomyEnvelope`、proposal和critique的
SHA-256。冻结门要求调用方提交阶段完成时捕获的四个预期哈希，任一输入在阶段完成后变化都会
fail closed。Research Design Agent 的预算账本、生产CLI、恢复编排和机器可执行统计schema已有
自动化测试；正式能力必须继续使用数据集无关合成夹具和独立验收，不能绑定历史数据集运行产物。

## 自主闭环工程基准

多被试合成夹具用于验证“画像—序贯搜索—锁定—独立确认”的科研周期效率，不构成真实 EEG 科学结论：

```powershell
.\.venv\Scripts\python.exe scripts\run_autonomous_cycle_benchmark.py `
  --spec experiments\specs\autonomous_cycle_benchmark.v3.json `
  --output artifacts\runs\autonomous-cycle-benchmark-v3\benchmark_result.json
```

v1 暴露候选全部饱和的问题，v2 暴露均值频谱被干扰节律误导和过早停止的问题，v3 增加分频带类别效应测量后再运行同一辨别性夹具。三轮结果与边界见 `docs/项目交接/AUTONOMOUS_CYCLE_BENCHMARK_2026-08-06.md`。

## 数据验收命令

官方数据保存在仓库外部，只读使用。安装最小数据依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[data,test]'
```

运行带标准语义 sidecar 的非BIDS数据验收：

```powershell
.\.venv\Scripts\python.exe -m bci_autodiscovery.profiling.semantic_validation `
  --dataset-root 'D:\path\to\dataset' `
  --semantics 'D:\path\to\dataset\dataset_semantics.json' `
  --output 'D:\path\to\dataset\dataset_validation.json'
```

验收器不修改源数据。输出包括全源 SHA-256、确定性容器检查、观察统计和机器可读限制。

## Research Design Agent v3

正式入口为 `bci-design-research`。它只接受冻结且通过独立 Dataset Critic 的
`DatasetLevelContract`，并要求 `AutonomyEnvelope` 精确绑定合同路径、SHA-256、付费币种和总预算。
Planner、Protocol Critic 与 Reviser 使用互相独立的模型运行时；它们只能读取标准画像与上游合同，
不能读取原始 EEG、实验结果或 frozen confirmation。

协议 schema v3 将以下内容冻结为机器可执行合同：权威 split-unit 目录、互斥且完备的数据角色、注册指标 ID、
聚合单元、缺失值政策、统计检验、置换次数、随机种子、alpha、多重比较、置信区间、成功/拒绝/不确定判定、
有限候选宇宙定义哈希、停止门和逐规则质量异常政策。文献前沿只保留为待方法工程验证的假设，不会被标记为有效方法。

每次模型调用先经过 append-only `budget_ledger.jsonl` 预检，返回后再按 provider 报告的
prompt/completion/cached tokens 和版本化价格记账。账本事件形成 SHA-256 链；恢复、重试、失败、科研周期、
候选执行、计算秒数和 confirmation access 使用同一计数体系。运行目录还包含连续 `audit.jsonl`、PID 状态、
可恢复 checkpoint、逐周期 proposal/critique 和最终 `frozen_protocol.json`。恢复命令拒绝仍存活或已终态的 run。

```powershell
.\.venv\Scripts\python.exe -m bci_autodiscovery.agents.research_design_cli `
  --dataset-level-contract artifacts\runs\<dataset-run>\dataset_level_contract.json `
  --autonomy-envelope artifacts\authorities\<envelope>.json `
  --provider kimi `
  --model kimi-k2.7-code `
  --pricing-currency CNY `
  --prompt-cost-per-million 6.5 `
  --completion-cost-per-million 27 `
  --cached-prompt-cost-per-million 1.3 `
  --pricing-source "Kimi official pricing snapshot YYYY-MM-DD"
```

中断恢复时把运行目录参数替换为 `--resume-run-dir artifacts\runs\<run-id>`，其余权威合同和价格参数保持不变。
旧的逐项人审入口已更名为 `bci-legacy-*`；其中协议注册/批准入口已移除，只保留历史工件和 registry 的只读检查。
