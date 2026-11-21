# LLM setup

## API key

Set the Gemini API key in the shell:

```powershell
$env:GEMINI_API_KEY="..."
```

Do not commit the key.

## Generate summaries

```powershell
python 03_Source_Code/llm/generate_summaries.py `
  --prompt faithful `
  --model gemini-3.5-flash-lite
```

To see the Gemini models available to your API key:

```powershell
python 03_Source_Code/llm/generate_summaries.py --list-models
```

This command uses the 100-article deterministic final sample and writes:

- `05_Predictions/llm_predictions.csv`
- `10_Documentation/llm_generation_log.json`

Generation through the API does not require a GPU.
