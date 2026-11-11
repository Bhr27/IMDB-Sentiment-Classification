# Dataset Source — Rotten Tomatoes (Cornell Movie Review)

**Dataset name:** Rotten Tomatoes / Cornell Movie Review Data (MR), Pang & Lee 2005

**Used here for TESTING ONLY** (no training / no fine-tuning was performed).

## Source links
- HuggingFace: https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes
- Original paper / page: https://www.cs.cornell.edu/people/pabo/movie-review-data/
- Citation: Bo Pang and Lillian Lee, "Seeing stars: Exploiting class relationships
  for sentiment categorization with respect to rating scales", ACL 2005.

## Files in this folder
| File | Rows | Purpose |
|------|------|---------|
| `rotten_tomatoes_full.csv` | 10,662 | Full dataset (train+val+test merged) — used entirely as a test set |
| `rotten_tomatoes_sample_1066.csv` | 1,066 | Random sample (seed=42), same size as the earlier IMDB-model test, for a fair comparison |
| `train.csv` / `validation.csv` / `test.csv` | 8530 / 1066 / 1066 | Original HuggingFace splits (kept for reference) |

## Notes
- Columns: `text`, `label` (0 = negative, 1 = positive).
- Perfectly class-balanced (50/50) in the full set.
- Loaded via the canonical namespaced id `cornell-movie-review-data/rotten_tomatoes`.
