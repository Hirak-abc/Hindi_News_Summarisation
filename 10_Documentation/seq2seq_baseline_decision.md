# Seq2Seq baseline decision

Agreed by Hirak (Seq2Seq owner) on DATE_HERE. Use these numbers as the Seq2Seq baseline for the comparison.

## Comparable results (full 8,847-example test set, same Unicode-aware scorer)

| Decoding | ROUGE-1 | ROUGE-2 | ROUGE-L | Source file |
|---|---:|---:|---:|---|
| Greedy | 0.237014 | 0.074970 | 0.203441 | seq2seq_greedy_rouge_unicode.csv |
| Repetition penalty 1.5 | 0.296962 | 0.080844 | 0.227799 | seq2seq_penalty_1.5_rouge_unicode.csv |
| Penalty 1.5 + trigram blocking | 0.297023 | 0.080860 | 0.227831 | seq2seq_full_trigram_penalty_1.5_rouge.csv |

Trigram blocking gives a negligible change from penalty 1.5 and is a separate decoding experiment.

## Excluded from the comparison

- seq2seq_final_report.txt and seq2seq_penalty_1.5_rouge.csv: scored with a regex tokenizer that splits Devanagari combining marks. Legacy, not comparable.
- 50-example and random-100-example results: different sample sizes and decoding settings.

## Scorer

All Hindi results use the Unicode-aware HindiTokenizer from 03_Source_Code/seq2seq/predict.py (rouge_score, use_stemmer=False). mT5 must use the same scorer.

## Verification

Anshu re-scored seq2seq_predictions_penalty_1.5.csv with this tokenizer and reproduced 0.296962 / 0.080844 / 0.227799 on 8,847 rows.
