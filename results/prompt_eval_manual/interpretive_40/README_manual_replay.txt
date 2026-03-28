Manual replay workflow

These files were generated for the first 40 items of: interpretive.

Files per dataset:
- <dataset>_prompts.csv: copy prompts out and paste manual responses into the manual_response column.
- <dataset>_replay_template.jsonl: final replay file for provider=replay; fill response_text for each item_id.

Run replay after filling the JSONL files:

python src/prompt_eval.py run --dataset interpretive --provider replay --model replay_manual --replay-file results/prompt_eval_manual/interpretive_replay_template.jsonl --run-id interpretive_replay_manual_40 --limit 40
python src/prompt_eval.py score --run-id interpretive_replay_manual_40
python src/prompt_eval.py aggregate --run-id interpretive_replay_manual_40
