# Risk Comovement Evaluation

- Records: `output/risk_exposure_sw3_430101_25_strict/03_dataset/risk_exposure_records.jsonl`
- Window: `2024-01-01` to `2024-12-31`
- Companies with risk vectors: 24
- Risk features: 107
- Usable pairs: 253
- Market factor: `sh000300`
- Similarity method: `cosine` over `risk_name` features

## Pair Similarity Signal

- Pearson(risk_similarity, market-neutral comovement): -0.0333
- Spearman(risk_similarity, market-neutral comovement): -0.0568
- Randomized Pearson mean: 0.0009
- Randomized Pearson p95: 0.1082
- Empirical p(random >= observed): 0.7120

## Similarity Bins

| bin | pairs | avg risk similarity | avg market-neutral comovement | avg raw comovement |
|---|---:|---:|---:|---:|
| low | 84 | 0.0000 | 0.4677 | 0.5532 |
| mid | 84 | 0.0000 | 0.4977 | 0.5663 |
| high | 85 | 0.0781 | 0.4954 | 0.5528 |

## Risk Groups

| group | companies | used pairs | market-neutral | raw |
|---|---:|---:|---:|---:|
