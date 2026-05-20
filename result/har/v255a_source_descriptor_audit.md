# v2.5.5a Source Structure Descriptor Audit

## Source Descriptors

| source | samples | classes | segments | roughness | boundary concentration | separability | within var | oracle best epochs |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `30TXT` | 106028 | 11 | 8 | 3.3992 | 1.0942 | 0.5627 | 2028217.8583 | 31TCJ:ep50, 32VNH:ep70, 33UVP:ep100 |
| `31TCJ` | 74156 | 12 | 8 | 2.9328 | 1.3325 | 0.4852 | 1181962.8998 | 30TXT:ep30, 32VNH:ep50, 33UVP:ep30 |
| `32VNH` | 53823 | 12 | 8 | 3.6746 | 1.1708 | 0.5603 | 3039542.5663 | 30TXT:ep70, 31TCJ:ep50, 33UVP:ep30 |
| `33UVP` | 43809 | 10 | 8 | 3.4150 | 1.2163 | 0.5261 | 1398463.4576 | 30TXT:ep70, 31TCJ:ep70, 32VNH:ep30 |

## Oracle Epoch Summary

| source | best-epoch counts |
|---|---|
| `30TXT` | ep50: 1, ep70: 1, ep100: 1 |
| `31TCJ` | ep30: 2, ep50: 1 |
| `32VNH` | ep30: 1, ep50: 1, ep70: 1 |
| `33UVP` | ep30: 1, ep70: 2 |

## Interpretation Guardrail

These descriptors are diagnostic only. Do not convert them into a training gate until they show a stable relationship with source checkpoint preference or downstream transfer behavior.
