# Risk Comovement Evaluation

- Records: `output/risk_exposure_sw3_430101_25_strict/03_dataset/risk_exposure_records.jsonl`
- Window: `2024-01-01` to `2024-12-31`
- Companies with risk vectors: 24
- Risk features: 118
- Usable pairs: 253
- Market factor: `sh000300`
- Similarity method: `cosine` over `hybrid` features

## Pair Similarity Signal

- Pearson(risk_similarity, market-neutral comovement): 0.0027
- Spearman(risk_similarity, market-neutral comovement): 0.0010
- Randomized Pearson mean: 0.0029
- Randomized Pearson p95: 0.0985
- Empirical p(random >= observed): 0.5020

## Similarity Bins

| bin | pairs | avg risk similarity | avg market-neutral comovement | avg raw comovement |
|---|---:|---:|---:|---:|
| low | 84 | 0.2254 | 0.4720 | 0.5478 |
| mid | 84 | 0.3296 | 0.5058 | 0.5740 |
| high | 85 | 0.4184 | 0.4831 | 0.5505 |

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
