# 自主科研周期工程基准：v1 → v3

> 日期：2026-08-06  
> 性质：合成数据工程验收；不得作为真实 EEG 科学结论或对外成果声明

## 1. 评价问题

在 8 个有限、完整、可执行候选中，画像引导的序贯策略能否比固定顺序和随机搜索用更少候选执行找到距离个体有限 oracle 不超过 `0.02 balanced_accuracy` 的 pipeline，并在独立合成会话上无重拟合确认？

四类异质夹具分别组合：

- 10 Hz / 20 Hz 个体目标节律；
- 类别特异空间分布 / 全局功率差异；
- 独立 profiling、search、confirmation 会话；
- confirmation 振幅衰减与频率漂移。

候选为 `bandpower_lda` 与 `csp_lda`，覆盖 `[4,40]`、`[8,13]`、`[8,30]`、`[13,30]` Hz。有限 oracle 只在事后用于评价策略，不参与引导排序或锁定。

## 2. 三轮结果

| 版本 | 诊断 | Guided 平均锁定周期 | Guided 近 oracle 锁定率 | Random 近 oracle 锁定率 |
|---|---|---:|---:|---:|
| v1 | 信号过强，所有候选饱和；Global Best 即所有个体 oracle，基准无辨别力 | 2.25 | 100% | 100% |
| v2 | 降低 SNR 并加入目标频带外干扰；均值频谱被干扰峰误导，且绝对分数停止规则过早锁定 | 3.00 | 25.0% | 50.25% |
| v3 | 数据与 oracle 不变；增加分频带类别效应测量，用类别相关证据选择目标频带 | 2.00 | 100% | 50.25% |

v3 详细结果：

- 16 名合成被试，4 种异质组；
- Guided 首次触达近 oracle：平均 1.1875 个科研周期；
- Random：平均 5.30125 个周期；
- 固定顺序：按“未在 8 个候选内触达”记为 9，平均 8.125 个周期；
- Guided 锁定时平均 oracle gap：0；
- Guided 平均确认 balanced accuracy：0.96875；
- search→confirmation 平均下降：0.018229；
- 最终锁定 4 种不同 pipeline，说明策略没有退化成单一 Global Best；
- 全局最佳固定 pipeline 的平均个体 oracle gap：0.135417；
- confirmation 侧没有任何重拟合。

## 3. 闭环发现

v1 和 v2 是必须保留的负结果。v2 的失败说明“知道个体主频”不能靠平均频谱峰替代：频谱峰可能属于与任务无关但能量更强的干扰。真正有用的下一项测量是分频带、类别相关的标准化效应。该测量已经进入通用 `measure_class_separability` 工具，而不是只写在基准脚本中的开发者结论。

因此这轮实验验证的是：

```text
失败结果
  → Result diagnosis
  → 识别画像观测缺口
  → 新增通用确定性测量
  → 画像引导策略重跑
  → 科研周期效率改善
```

这符合项目要求的 Agent/工具闭环，但尚未证明当前 LLM Agent 在真实 EEG 上能自主完成同样的诊断。

## 4. 可复现产物

- v1：`artifacts/runs/autonomous-cycle-benchmark-v1/benchmark_result.json`
- v2：`artifacts/runs/autonomous-cycle-benchmark-v2/benchmark_result.json`
- v3：`artifacts/runs/autonomous-cycle-benchmark-v3/benchmark_result.json`
- v3 重放：`artifacts/runs/autonomous-cycle-benchmark-v3/benchmark_result.replay.json`
- v3 原始与重放 SHA-256 均为 `310081DF2089F719C57534543106B65FA7C83CC4CE926BBAD02E624D48273B77`
- v3 规格：`experiments/specs/autonomous_cycle_benchmark.v3.json`
- 运行入口：`scripts/run_autonomous_cycle_benchmark.py`

## 5. 不可外推的边界

- 合成生成机制只覆盖两个频率和两类信号模式；
- 候选只有两个模型家族，不是完整 EEG pipeline 搜索空间；
- Guided 策略是透明确定性策略，尚不是付费模型或真实知识检索驱动的 Agent 运行；
- 未执行任何真实数据集的多被试冻结确认；
- 未建立置信区间、统计显著性或跨数据集复现；
- 因此当前结论仅为“工程闭环和效率评价方法可运行”，不是“系统已经优于研究者或 AutoML”。
