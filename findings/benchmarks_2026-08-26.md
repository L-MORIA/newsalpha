# Benchmarks: STTM vs alternatives — 2026-08-26

## Дизайн
- Период: 2015-01-02 to 2021-12-31 (366 weeks)
- Universe: 39 TQBR tickers
- Стратегия: top-20% equal-weight, gross (no transaction costs)
- Бэктест: `newsalpha.backtest.portfolio.backtest()`

## Результаты

| Стратегия | Sharpe | Ann Return | MaxDD |
|---|---|---|---|
| STTM (news-based) | **1.217** | 21.3% | -22.0% |
| Equal-weight | 1.069 | 16.0% | -19.2% |
| Random Top-20% (mean) | 0.940 ± 0.195 | 16.3% | -23.1% |
| Random 95% CI | [0.529, 1.271] | — | — |
| Momentum 4w | 0.895 | 15.5% | -24.6% |
| Momentum 12w | 0.879 | 15.6% | -20.5% |
| Contrarian 4w | 0.824 | 15.9% | -19.5% |
| Random walk signal | 0.892 | 14.7% | -21.1% |

## Ключевые выводы

1. **STTM vs Random: z = 1.42, p = 0.078** — не значимо на уровне 5%
2. **Equal-weight (1.069) лучше Momentum (0.895)** — на MOEX 2015-2021 momentum отрицательный
3. **STTM = Equal-weight + 0.148** — небольшой selection bias, не альфа
4. **Random walk даёт Sharpe 0.89** — даже случайный шум создаёт profit на растущем рынке

## Вывод
Long-only Sharpe 1.217 ≈ Equal-weight (1.069) + micro-selection bias. Новостной сигнал не даёт значимого прироста.
