# Agent 输出区

这里保存各科研 Agent 面向用户发布的派生视图，按数据集和研究阶段组织：

```text
agent_outputs/<dataset_id>/
├── 01_dataset_understanding/  # 数据集画像、认知空间、冻结合同
├── 02_research_design/        # 冻结研究协议
├── 03_method_capabilities/    # 方法工程 Agent 验证过的有限能力
├── 04_subject_profiles/       # 被试级画像
├── 05_pipeline_search/        # 搜索过程、候选与 pipeline lock
├── 06_confirmation/           # 一次性冻结确认结果
└── 07_reports/                # 内部证据报告与科学 Critic 结论
```

这里不是审计原件库。每份发布文件都在数据集目录的 `index.json` 中记录来源
`run_id`、原始绝对路径和 SHA-256；权威、不可变的运行产物仍位于
`artifacts/runs/<run_id>/`。

`03_method_capabilities/` 只有在方法工程 Agent 完成许可证、接口、泄漏测试和最小
复现实验后才会写入，不能把静态候选组件表冒充为已支持能力。
