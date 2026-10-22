# Data audit

## Dialect identification (IADD)

- raw rows: 135804
- `general` removed: 2500
- empty removed: 11
- too long removed: 144
- conflicting texts dropped: 301
- duplicates removed: 8098
- classes dropped (too small): {'Iraqi': 212}
- final rows: 123009

| label | rows | sources |
|---|---|---|
| Levantine | 84157 | {'SHAMI': 66003, 'PADIC': 11440, 'AOC': 6323, 'DART': 391} |
| Maghrebi | 27689 | {'PADIC': 16599, 'TSAC': 10734, 'DART': 328, 'AOC': 28} |
| Gulf | 6620 | {'AOC': 6200, 'DART': 420} |
| Egyptian | 4543 | {'AOC': 4124, 'DART': 419} |

> If a class comes almost entirely from one source, a model can learn the source's style
> instead of the dialect. Report cross-source results and review errors by source.

## Egyptian -> MSA (EGY_MSA_Translation)

- raw rows: {'train': 57000, 'valid': 100, 'test': 2227}
- after dedupe: {'train': 55162, 'valid': 100, 'test': 2223}
- gold rows checked for leakage: 0
- valid removed (overlap with test/gold): 0
- train removed (leakage): 463
- final rows: {'train': 54699, 'valid': 100, 'test': 2223}
- identical EGY==MSA share: {'train': 0.0063, 'valid': 0.01, 'test': 0.0103}
