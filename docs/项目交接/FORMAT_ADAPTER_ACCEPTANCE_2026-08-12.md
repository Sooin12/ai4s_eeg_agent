# 多格式语义适配器独立验收

> 日期：2026-08-12；2026-08-14 增补通用 MAT epoch reader
> 验收范围：新数据集的“格式识别 → 适配器检索 → 确定性验证 → DatasetProfile”链路  
> 结论：**通过**

## 交付能力

### 1. 无 sidecar 的标准组织

合法 BIDS-EEG 目录继续由 `bids_eeg_v1` 处理，支持 BIDS 规定的 EDF/BDF、
BrainVision Core 和 EEGLAB 信号容器。

### 2. 非 BIDS、MNE 可读的主流格式

`mne_raw_semantic_v1` 覆盖 EDF/EDF+、BDF/BDF+、BrainVision Core、EEGLAB、
FIF、GDF、Neuroscan CNT 和 EGI MFF。运行要求：

- 数据根目录包含 `dataset_semantics.json`；
- 语义合同明确被试/会话规则、annotation 映射和完整标准画像；
- 先运行 `bci-validate-semantic-dataset`；
- MNE 本地读取的采样率、通道、EOG、工频、annotations、被试/会话/run计数
  必须与合同一致；
- 所有源文件及 BrainVision/EEGLAB 配套二进制文件均绑定 SHA-256。

### 3. MATLAB epoch 数组

`mat_epoch_semantic_v1` 面向任意三维 epoch MAT 数组，不依赖数据集 ID 或私有目录。
sidecar 声明对象名、轴顺序、被试/会话/可选 run 正则、事件码、采样率、通道与单位；
本地确定性 reader 独立核验：

- 每个源的对象、形状、轴和 label 数；
- 被试/会话/run 身份唯一性和缺失 subject-session 网格；
- 事件码映射、类别与 trial 数；
- 非有限值、全平通道、trial-count anomalies；
- 全部源文件 SHA-256。

标准 `DatasetProfile` 的 subjects、sessions、runs、trials、class counts、epoch 时长和
质量计数由观察值生成，不再要求开发者预填。

### 4. 其他灵活容器

`declarative_semantic_v1` 覆盖 XDF、NWB、通用 HDF5、NPY/NPZ 和暂未安装本地
读取器的 BVRF。合同必须明确：

- `signal_object_or_stream`；
- `axis_order`，至少包含 `channel` 和 `sample`；
- `event_source` 与 `event_code_mapping`；
- `data_role_policy`；
- 完整标准 `DatasetProfile`。

该路径不推断容器内部语义。当前验证范围是合同schema、标准画像合同和全部源文件
SHA绑定；若要对任意容器内部对象做独立数值核验，需再安装对应本地读取库或添加
确定性 reader。此限制会写入验证产物。

## 单独验收用例

| 用例 | 预期 | 结果 |
|---|---|---|
| 真实 FIF 文件，MNE读取2通道、100 Hz和L/R annotations | 自动选择 `mne_raw_semantic_v1` 并生成画像 | 通过 |
| FIF只含一个事件但合同声明两个事件 | 验证阶段拒绝 | 通过 |
| NPZ声明数组轴、事件源和角色政策 | 自动选择 `declarative_semantic_v1` 并生成画像 | 通过 |
| 两被试×两会话 MAT，身份为非数字标签，1 run少1 trial | 通用 reader 自动生成计数和异常画像 | 通过 |
| MAT subject-session 网格缺一个单元 | 画像明确记录缺失单元 | 通过 |
| 验证后修改NPZ源文件 | SHA复核失败，退回语义映射状态 | 通过 |
| declarative合同缺少数据角色政策 | schema验证拒绝 | 通过 |
| 无sidecar的普通MAT | 只能识别格式，不能生成画像 | 通过 |
| 验证清单被改为空源列表 | 适配器拒绝选择 | 通过 |

独立验收命令：

```powershell
python -m pytest tests/unit/test_semantic_dataset_adapters.py `
  tests/unit/test_dataset_format_catalog.py `
  tests/unit/test_dataset_profiler_agent.py -q
```

验收结果以当前测试输出为准；旧的固定计数不再作为能力声明。

## 新数据使用方式

将语义合同保存为数据根目录下的 `dataset_semantics.json`，schema参考：

`configs/dataset_semantics.schema.v1.json`

然后执行：

```powershell
python -m bci_autodiscovery.profiling.semantic_validation `
  --dataset-root D:\path\to\dataset `
  --semantics D:\path\to\dataset\dataset_semantics.json `
  --output D:\path\to\dataset_validation.json
```

正式 Dataset-Level Agent 使用相同的 `--dataset-root` 和 `--validation`，会先搜索
已注册适配器，只有验证产物、sidecar和源文件三者哈希一致时才进入画像阶段。

## 边界

- 原始信号不进入LLM上下文；
- 格式扩展名不决定轴、事件或研究数据角色；
- sidecar不是随意说明文本，而是被schema、标准画像验证器和源SHA约束的机器合同；
- 格式读取能力不等于某种科学方法已经由 Method Engineering Agent 验收。
