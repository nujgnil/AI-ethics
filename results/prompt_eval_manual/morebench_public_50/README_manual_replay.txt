Manual replay workflow

These files were generated for the first 50 items of: morebench_public.

Files per dataset:
- <dataset>_prompts.csv: copy prompts out and paste manual responses into the manual_response column.
- <dataset>_replay_template.jsonl: final replay file for provider=replay; fill response_text for each item_id.

Run replay after filling the JSONL files:

python src/prompt_eval.py run --dataset morebench_public --provider replay --model replay_manual --replay-file results/prompt_eval_manual/morebench_public_replay_template.jsonl --run-id morebench_public_replay_manual_50 --limit 50
python src/prompt_eval.py score --run-id morebench_public_replay_manual_50
python src/prompt_eval.py aggregate --run-id morebench_public_replay_manual_50
