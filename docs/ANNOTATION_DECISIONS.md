# Annotation decisions

Generated from `config/annotation_decisions.yaml`. Edit that file after the professor replies, then rerun `python scripts/build_dataset.py --annotations official`.

The official rule is `elapsed = clock annotation − DebutTrace`, with midnight rollover when needed. Raw workbooks are never modified.

## Applied decisions

- 230406B_A: duplicate rows scored; selected row 11 (score=0.189095, ratio=2.184).
- 230515B_G: duplicate rows scored; selected row 20 (score=0.354451, ratio=4.076).
- 210504B_C CE1: proposed end 12:51:06 scored 0.305716, ratio=2.610; correction retained.
- 210204B_C: missing from official workbook; provisional intervals retained as fallback.
- 210208B_G: missing from official workbook; provisional intervals retained as fallback.
- 210427B_C: official DebutTrace places the interval at 1031.000–1049.000 s; score=0.306318, ratio=2.695; included.

## Signal scores used by configurable decisions

### 230406B_A

| Source row | Score | Inside/outside ratio | Inside windows |
|---:|---:|---:|---:|
| 11 | 0.189095 | 2.184 | 178 |
| 19 | 0.173201 | 2.086 | 202 |

### 230515B_G

| Source row | Score | Inside/outside ratio | Inside windows |
|---:|---:|---:|---:|
| 12 | 0.336547 | 3.996 | 185 |
| 20 | 0.354451 | 4.076 | 136 |

### 210504B_C_correction

| Source row | Score | Inside/outside ratio | Inside windows |
|---:|---:|---:|---:|
| — | 0.305716 | 2.610 | 29 |

### 210427B_C_official_validation

| Source row | Score | Inside/outside ratio | Inside windows |
|---:|---:|---:|---:|
| — | 0.306318 | 2.695 | 73 |

## Exclusions

- `200625A_F`: No annotations; status remains unknown rather than confirmed seizure-free.

Records marked `fallback` in `data/interim/official/offsets.csv` use provisional intervals because they are absent from the updated workbook. All other included rows use method `official`.
