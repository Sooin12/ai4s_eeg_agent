# EEG 数据格式适配器调研与落地

> 日期：2026-08-12  
> 状态：格式识别目录、MNE语义适配器、MAT epoch reader 和declarative sidecar适配器已实现

## 结论

数据接入必须拆为两层：

1. **格式识别层**只根据目录结构、扩展名和必需配套文件确认容器候选；
2. **语义适配层**必须用确定性程序确认被试、会话、信号对象、数组轴、采样率、
   通道、单位、事件含义、时间零点和数据角色，之后才能输出标准 `DatasetProfile`。

`.mat`、`.h5`、`.npz` 只是通用容器。系统禁止从扩展名猜测
`channel × sample × trial`、事件码含义或 search/confirmation 角色。

## 主流格式覆盖

| 类别 | 已识别格式 | 当前接入状态 |
|---|---|---|
| 标准组织 | BIDS-EEG | `bids_eeg_v1` 可在具备验证产物时生成画像 |
| BIDS允许的信号容器 | EDF/EDF+、BDF/BDF+、BrainVision VHDR/VMRK/EEG、EEGLAB SET/FDT | 可识别；位于合法BIDS结构时由BIDS适配器处理 |
| 常见科研/厂商格式 | FIF、GDF、Neuroscan CNT、EGI MFF | `mne_raw_semantic_v1`本地核验头信息和annotations |
| 多流与通用神经科学容器 | XDF、NWB | `declarative_semantic_v1`强制显式选择流/series和事件映射 |
| MATLAB epoch 数组 | 三维 MAT epochs + labels | `mat_epoch_semantic_v1`本地核验数组、身份、事件、质量与计数 |
| 其他通用数组容器 | FieldTrip候选、HDF5、NPY/NPZ | `declarative_semantic_v1`强制显式数组轴、事件与角色合同 |
| 新版Brain Products | BVRF（BVRH/BVRM/BVRD） | 当前走declarative合同；安装确定性reader后可升级为内部核验 |

## 实现位置

- 格式探测实现：`src/bci_autodiscovery/profiling/formats.py`
- 画像适配器注册：`src/bci_autodiscovery/profiling/adapters.py`
- 语义合同验证：`src/bci_autodiscovery/profiling/semantic_validation.py`
- 版本化政策和最小sidecar字段：`configs/dataset_adapter_catalog.v1.json`
- sidecar JSON Schema：`configs/dataset_semantics.schema.v1.json`

当识别到格式、但没有可生成画像的语义适配器时，检查结果是
`recognized_format_requires_semantic_mapping`，不会把“识别成功”误报为“画像完成”。

## 官方依据

- BIDS-EEG：<https://bids-specification.readthedocs.io/en/stable/modality-specific-files/electroencephalography.html>
- MNE 支持格式：<https://mne.tools/stable/documentation/implementation.html#supported-data-formats>
- EDF/EDF+：<https://www.edfplus.info/specs/index.html>
- BrainVision Core：<https://www.brainproducts.com/support-resources/brainvision-core-data-format-1-0/>
- BrainVision Recording：<https://www.brainproducts.com/support-resources/brainvision-recording-format/>
- EEGLAB：<https://eeglab.org/tutorials/ConceptsGuide/Data_Structures.html>
- XDF：<https://github.com/sccn/xdf/wiki/Specifications>
- NWB：<https://nwb-schema.readthedocs.io/en/stable/format.html>
