# Risk Comovement Evaluation

- Records: `output/risk_exposure_sw3_430101_25_strict/03_dataset/risk_exposure_records.jsonl`
- Window: `2024-01-01` to `2024-12-31`
- Companies with risk vectors: 24
- Risk features: 9
- Usable pairs: 253
- Market factor: `sh000300`
- Similarity method: `cosine` over `category` features

## Pair Similarity Signal

- Pearson(risk_similarity, market-neutral comovement): 0.0286
- Spearman(risk_similarity, market-neutral comovement): 0.0289
- Randomized Pearson mean: 0.0014
- Randomized Pearson p95: 0.1047
- Empirical p(random >= observed): 0.3340

## Similarity Bins

| bin | pairs | avg risk similarity | avg market-neutral comovement | avg raw comovement |
|---|---:|---:|---:|---:|
| low | 84 | 0.4698 | 0.4827 | 0.5570 |
| mid | 84 | 0.6889 | 0.4867 | 0.5557 |
| high | 85 | 0.8315 | 0.4914 | 0.5595 |

## Risk Groups

| group | companies | used pairs | market-neutral | raw |
|---|---:|---:|---:|---:|
| 汇率利率 | 2 | 1 | 0.6365 | 0.7345 |
| 市场价格 | 14 | 91 | 0.5312 | 0.6144 |
| 需求周期 | 20 | 190 | 0.5135 | 0.5892 |
| 财务流动性 | 21 | 210 | 0.4951 | 0.5690 |
| 供应链 | 15 | 105 | 0.4940 | 0.5568 |
| 客户信用/集中度 | 11 | 55 | 0.4457 | 0.5106 |
| 政策监管 | 5 | 10 | 0.4355 | 0.4939 |
| 诉讼合规治理 | 5 | 10 | 0.4081 | 0.4653 |
