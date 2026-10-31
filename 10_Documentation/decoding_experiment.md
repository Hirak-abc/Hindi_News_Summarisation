# Decoding experiment (no retraining)

The existing checkpoint was loaded on CPU. No training was run. The checkpoint and original prediction files were preserved.

## Decoding changes

`03_Source_Code/seq2seq/predict.py` supports per-summary no-repeat n-gram blocking. The default n-gram size is 3; `--no-repeat-ngram-size 0` disables it. EOS is exempt from blocking. Repetition penalty remains configurable and defaults to 1.5.

Minimum length defaults to 0, so EOS is not delayed. Maximum generation length remains 100 tokens. No additional loop detector was added. New prediction and score filenames avoid overwriting the existing outputs.

The ROUGE tokenizer was also corrected to keep Unicode letters, combining marks, and numbers together. The former regex split Devanagari combining marks, which inflated the earlier ROUGE scores. The original score files were not overwritten; corrected baseline scores are saved separately.

## Full-test results

All 8,847 test articles were decoded on CPU with penalty 1.5 and trigram blocking. The same corrected Hindi-aware tokenizer was used to evaluate the old greedy predictions, old penalty-1.5 predictions, and new predictions.

| Decoding | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 |
|---|---:|---:|---:|
| Existing greedy predictions | 0.237014 | 0.074970 | 0.203441 |
| Existing penalty 1.5 predictions | 0.296962 | 0.080844 | 0.227799 |
| Penalty 1.5 + trigram blocking | 0.297023 | 0.080860 | 0.227831 |

Penalty 1.5 improved all three corrected full-test ROUGE F1 values over greedy decoding. Adding trigram blocking changed only 23 of 8,847 summaries. It eliminated the 23 summaries with a repeated model-token trigram; the new output had none. The corresponding ROUGE gains over penalty 1.5 were very small: +0.000061 ROUGE-1, +0.000016 ROUGE-2, and +0.000032 ROUGE-L.

The new output averages 18.44 model tokens; 11 summaries are under five tokens, the longest has 58 tokens, and 2,192 summaries contain `<unk>`. These safeguards do not prevent single-word repetition, incoherence, hallucinations, or unknown tokens.

## Sample comparison and manual review

The first 50 rows were used for a smoke test. A separate deterministic random sample of 100 articles (seed 42) was used to compare penalty values. Full-test inference was then run once with penalty 1.5 and trigram blocking.

Corrected ROUGE F1 on the random sample:

| Decoding | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|
| Existing greedy predictions | 0.239049 | 0.080553 | 0.206203 |
| Existing penalty 1.5 predictions | 0.288711 | 0.080270 | 0.221690 |
| Penalty 1.0 + trigram blocking | 0.268008 | 0.090558 | 0.228478 |
| Penalty 1.2 + trigram blocking | 0.285572 | 0.080815 | 0.229448 |
| Penalty 1.5 + trigram blocking | 0.288711 | 0.080270 | 0.221690 |

Penalty 1.5 + trigram blocking matched the old penalty-1.5 output on the random sample. Manual inspection of six random full-test examples found a mix: some captured the broad topic, while others were factually wrong or unrelated. For example, a summary about a cricket board's response misstated the board's role, a bird-noise article was summarized as a claim about mammals, and an Urdu-newspaper roundup was described as being about COVID-19. This small manual check is illustrative, not a formal human evaluation.

## Recommendation and output files

Keep repetition penalty 1.5 as the stronger tested baseline. Trigram blocking is a reasonable optional guard against exact repeated trigrams, but the full-test score change is negligible and it does not fix the main coherence and factuality problems. Do not claim a meaningful summary-quality improvement from trigram blocking. Within the no-retraining constraint, the current evidence does not justify another decoding change; a more substantial quality improvement may require changes outside inference-time decoding.

Full-test predictions and corrected ROUGE:

- [Full-test predictions with trigram blocking](../05_Predictions/seq2seq_predictions_full_trigram_penalty_1.5.csv)
- [Full-test corrected ROUGE](../06_Evaluation_Results/seq2seq_full_trigram_penalty_1.5_rouge.csv)
- [Corrected ROUGE for existing penalty-1.5 predictions](../06_Evaluation_Results/seq2seq_penalty_1.5_rouge_unicode.csv)
- [Corrected ROUGE for existing greedy predictions](../06_Evaluation_Results/seq2seq_greedy_rouge_unicode.csv)

Random-100 predictions and corrected scores are saved as `seq2seq_predictions_random100_penalty_{1.0,1.2,1.5}_trigram.csv` and `seq2seq_random100_penalty_{1.0,1.2,1.5}_trigram_rouge.csv` in their respective output folders. The original prediction and score files remain unchanged.
