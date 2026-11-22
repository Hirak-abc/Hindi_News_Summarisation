# TextRank run notes

## Method

The extractive TextRank implementation uses:

- Hindi sentence splitting on `।`, `!`, `?`, and `.`
- Character n-grams from 2 to 4 as sentence features
- Cosine similarity between TF-IDF sentence vectors
- PageRank to rank sentences
- Original article order for the final summary
- A maximum of 3 sentences and 100 words

## Common evaluation

Predictions were generated for all 8,847 articles in `01_Dataset/processed/common_test_set.csv`.

Configuration:

- Maximum sentences: 3
- Maximum words: 100
- Random seed: not applicable
- Scorer: Unicode-aware Hindi tokenizer
- Stemmer: disabled

Results:

| Metric | F1 |
|---|---:|
| ROUGE-1 | 0.206809 |
| ROUGE-2 | 0.052218 |
| ROUGE-L | 0.137857 |

Outputs:

- `05_Predictions/textrank_predictions.csv`
- `06_Evaluation_Results/textrank_rouge_scores.csv`
