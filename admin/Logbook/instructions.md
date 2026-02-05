1. Set up environment (use **venv** so it matches [start.ps1](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/Ling%20Jun/.vscode/extensions/openai.chatgpt-0.4.69-win32-x64/webview/# "start.ps1")):

`<span>.\setup.ps1 -VenvPath ".\venv" </span>`

2. Activate env:

`<span>.\venv\Scripts\Activate.ps1 </span>`

3. Run preprocessing pipeline:

`<span>python scripts\preprocessing\run_preprocess_all.py </span>`

4. Run baseline models first (ETHICS + NormBank):

`<span>python -m src.model_train --dataset ethics --model tfidf_logreg python -m src.model_train --dataset ethics --model tfidf_linearsvc python -m src.model_train --dataset normbank --model tfidf_logreg python -m src.model_train --dataset normbank --model bow_mnb </span>`

5. Run transformer pilots (small sample first):

`<span>python -m src.model_train --dataset ethics --model distilbert-base-uncased --max-train-samples 5000 --max-test-samples 2000 --epochs 1 python -m src.model_train --dataset normbank --model distilbert-base-uncased --max-train-samples 5000 --max-test-samples 2000 --epochs 1 </span>`

6. Run full allocation batch for currently trainable datasets:

`<span>python -m src.run_allocations --datasets ethics normbank --epochs 2 </span>`

7. Check outputs:
   * [metrics.csv](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/Ling%20Jun/.vscode/extensions/openai.chatgpt-0.4.69-win32-x64/webview/# "results/metrics.csv")
   * [ualitative_examples.txt](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/Ling%20Jun/.vscode/extensions/openai.chatgpt-0.4.69-win32-x64/webview/# "results/ualitative_examples.txt")
8. Then do the remaining blockers:

* Add MFRC label mapping (currently no supervised **label** column).
* Keep MoralBench as eval/prompt-only (or add labels/splits if you want supervised use).
* Add Delphi data if you want to run its allocated models.

If you want, I can give you the exact next 1-week execution plan (what to run each day).
