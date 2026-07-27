# Priority scheduling sweep (Qwen3-1.7B, 3060 Ti)

tiers: P2 premium 10% / P1 standard 30% / P0 batch 60%; prompts=400, max_tokens=128

| rate | scheduler | tier | TTFT p50 | TTFT p99 | TTFT max | ITL p50 | thrpt tok/s |
|------|-----------|------|----------|----------|----------|---------|-------------|
| 8.0 | baseline | P2-prem | 37.2 | 141.0 | 146.2 | 16.7 | 1005.4 |
| 8.0 | baseline | P1-std | 40.8 | 164.2 | 165.7 | 16.7 | 1005.4 |
| 8.0 | baseline | P0-batch | 56.1 | 199.9 | 204.1 | 16.7 | 1005.4 |
| 8.0 | priority | P2-prem | 37.2 | 146.4 | 147.0 | 16.7 | 1005.1 |
| 8.0 | priority | P1-std | 40.9 | 164.0 | 165.7 | 16.8 | 1005.1 |
| 8.0 | priority | P0-batch | 42.0 | 201.6 | 206.1 | 16.8 | 1005.1 |
| 14.0 | baseline | P2-prem | 3402.0 | 5494.4 | 5536.9 | 20.7 | 1388.7 |
| 14.0 | baseline | P1-std | 4191.8 | 5873.9 | 6478.2 | 20.7 | 1388.7 |
| 14.0 | baseline | P0-batch | 3919.7 | 5907.5 | 7457.1 | 20.8 | 1388.7 |
| 14.0 | priority | P2-prem | 108.1 | 1171.4 | 1180.2 | 20.7 | 1352.0 |
| 14.0 | priority | P1-std | 4052.9 | 5366.7 | 5420.9 | 20.8 | 1352.0 |
| 14.0 | priority | P0-batch | 5217.6 | 8386.4 | 8581.6 | 20.7 | 1352.0 |
| 24.0 | baseline | P2-prem | 6512.9 | 15540.8 | 15545.7 | 20.7 | 1420.7 |
| 24.0 | baseline | P1-std | 10814.0 | 15916.9 | 15924.9 | 20.7 | 1420.7 |
| 24.0 | baseline | P0-batch | 8771.3 | 15920.2 | 15985.7 | 20.7 | 1420.7 |
| 24.0 | priority | P2-prem | 204.1 | 16233.2 | 16620.2 | 20.6 | 1399.2 |
| 24.0 | priority | P1-std | 11111.4 | 16816.9 | 16995.3 | 20.7 | 1399.2 |
| 24.0 | priority | P0-batch | 10137.4 | 16826.8 | 18878.8 | 20.7 | 1399.2 |
| 24.0 | no-aging | P2-prem | 193.2 | 2804.9 | 2808.6 | 20.6 | 1414.2 |
| 24.0 | no-aging | P1-std | 1354.3 | 3152.2 | 3173.3 | 20.7 | 1414.2 |
| 24.0 | no-aging | P0-batch | 15176.4 | 18246.8 | 18459.3 | 20.7 | 1414.2 |

## Headline

- rate 8.0: **P2 premium p99 TTFT 141 -> 146 ms (+3.8%)** (baseline FCFS -> priority+aging)
- rate 14.0: **P2 premium p99 TTFT 5494 -> 1171 ms (-78.7%)** (baseline FCFS -> priority+aging)
- rate 24.0: **P2 premium p99 TTFT 15541 -> 16233 ms (+4.5%)** (baseline FCFS -> priority+aging)
- rate 24.0: **P0 batch max-TTFT** aging 18879 ms vs no-aging 18459 ms (starvation bound)