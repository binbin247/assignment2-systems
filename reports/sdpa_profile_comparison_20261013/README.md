# SDPA profiling 对比报告

- `report.md`：中文报告正文与统计表。
- `report.html`：可独立打开的图表报告。
- `analysis.json`、五份 CSV：分析结果与逐次数据。
- `evidence/`：官方导出、原始配置和代码快照；较早手写原始报告已单独保留。
- `sqlite/`：三份只读导出数据库，可用于复现。

复现统计：

```bash
python3 analyze_profiles.py
```

生成文本与 reviewed snapshot：

```bash
python3 compose_report.py
```

HTML 使用本地 Data app 已验证的构建导出，分析脚本不运行模型、不重新采集 GPU。

本次报告只分析 forward；性能倍数受跨次采集环境影响，见正文限定。
