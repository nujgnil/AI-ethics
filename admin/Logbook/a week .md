**1-Week Execution Plan (Feb 5-11, 2026)**

1. **Thu, Feb 5, 2026: Environment + Data Freeze**

* Run:

`<span>.\setup.ps1 -VenvPath ".\venv" .\venv\Scripts\Activate.ps1 python scripts\preprocessing\run_preprocess_all.py </span>`

* Verify files exist and are non-empty:
  * [ethics.csv](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/Ling%20Jun/.vscode/extensions/openai.chatgpt-0.4.69-win32-x64/webview/# "Data/processed/ethics/ethics.csv")
  * [normbank.csv](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/Ling%20Jun/.vscode/extensions/openai.chatgpt-0.4.69-win32-x64/webview/# "Data/processed/normbank/normbank.csv")
  * [processed_dataset_summary.csv](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/Ling%20Jun/.vscode/extensions/openai.chatgpt-0.4.69-win32-x64/webview/# "Data/eda/processed/processed_dataset_summary.csv")
* Done criteria: preprocessing completes with no crash.

2. **Fri, Feb 6, 2026: Baseline Models (Fast)**

* Run:

`<span>python -m src.model_train --dataset ethics --model tfidf_logreg python -m src.model_train --dataset ethics --model tfidf_linearsvc python -m src.model_train --dataset normbank --model tfidf_logreg python -m src.model_train --dataset normbank --model bow_mnb </span>`

* Done criteria: 4 rows appended to [metrics.csv](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/Ling%20Jun/.vscode/extensions/openai.chatgpt-0.4.69-win32-x64/webview/# "results/metrics.csv").

3. **Sat, Feb 7, 2026: Transformer Pilot (Small)**

* Run:

`<span>python -m src.model_train --dataset ethics --model distilbert-base-uncased --max-train-samples 5000 --max-test-samples 2000 --epochs 1 python -m src.model_train --dataset ethics --model bert-base-uncased --max-train-samples 5000 --max-test-samples 2000 --epochs 1 python -m src.model_train --dataset normbank --model distilbert-base-uncased --max-train-samples 5000 --max-test-samples 2000 --epochs 1 </span>`

* Done criteria: model downloads work, metrics written, no OOM.

4. **Sun, Feb 8, 2026: Main Transformer Runs**

* Run:

`<span>python -m src.model_train --dataset ethics --model roberta-base --epochs 2 python -m src.model_train --dataset ethics --model microsoft/deberta-v3-base --epochs 2 python -m src.model_train --dataset normbank --model roberta-base --epochs 2 </span>`

* Done criteria: full-size runs complete (or reduced with sample flags if compute limits).

5. **Mon, Feb 9, 2026: Allocation Sweep + Gaps Log**

* Run:

`<span>python -m src.run_allocations --datasets ethics normbank --epochs 2 </span>`

* Review skipped reasons for **mfrc** and **moralbench**.
* Done criteria: one consolidated run log and metrics table ready.

6. **Tue, Feb 10, 2026: Error Analysis + Calibration Notes**
   * Review:
     * [metrics.csv](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/Ling%20Jun/.vscode/extensions/openai.chatgpt-0.4.69-win32-x64/webview/# "results/metrics.csv")
     * [ualitative_examples.txt](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/Ling%20Jun/.vscode/extensions/openai.chatgpt-0.4.69-win32-x64/webview/# "results/ualitative_examples.txt")
   * Produce a short findings note in **admin/Logbook/**:
     * Best model per dataset
     * Worst failure pattern
     * Calibration observations (ECE/Brier)
   * Done criteria: written summary you can reuse in report.
7. **Wed, Feb 11, 2026: Cleanup + Next Sprint Setup**

* Create/refresh:
  * [scripts.md](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/Ling%20Jun/.vscode/extensions/openai.chatgpt-0.4.69-win32-x64/webview/# "admin/Logbook/scripts.md") (already done, just verify)
  * new log entry for this week’s runs
* Define next sprint tasks:
  * MFRC label mapping implementation
  * MoralBench evaluation protocol (prompt-only)
  * Delphi ingestion
* Done criteria: repo state is reproducible and next tasks are explicit.

If you want, I can turn this into a checklist file ([week-plan-2026-02-05-to-2026-02-11.md](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/Ling%20Jun/.vscode/extensions/openai.chatgpt-0.4.69-win32-x64/webview/# "admin/Logbook/week-plan-2026-02-05-to-2026-02-11.md")) and prefill command blocks for copy-paste.
