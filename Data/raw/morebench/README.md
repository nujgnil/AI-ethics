## MoReBench Setup Note

Date checked: 2026-03-16

Official sources:

- Project page: https://morebench.github.io/
- GitHub: https://github.com/MoReBench/MoReBench
- Hugging Face dataset page: https://huggingface.co/datasets/morebench/morebench

Expected local raw files for this repo:

- `Data/raw/morebench/morebench_public.csv`
- `Data/raw/morebench/morebench_theory.csv`

What has been done:

- Verified that the repo did not contain the raw MoReBench CSV files.
- Standardized the expected local location to `Data/raw/morebench/`.
- Updated the local preprocess scripts to look in `Data/raw/morebench/` first, and fall back to the old root-level CSV paths if needed.
- Added a downloader entrypoint at `src/download_morebench.py`.
- Corrected the downloader after verifying that the old `kellycyy/morebench` example is stale; the current public Hugging Face dataset repo is `morebench/morebench`.

Notes:

- The public subset and theory subset are treated as separate inputs in this repo.
- The official Hugging Face page exposes both CSV files in the dataset files list.
- If you download manually, keep the filenames exactly as `morebench_public.csv` and `morebench_theory.csv`.

Suggested next step after downloading:

- Run the matching preprocess notebook or script for each file:
  - `src/preprocess/preprocess_morebench_public.py`
  - `src/preprocess/preprocess_morebench_theory.py`

Download command:

```powershell
python src\download_morebench.py
```

Optional flags:

- `--subset public`
- `--subset theory`
- `--overwrite`

Citation

@misc{chiu2025morebenchevaluatingproceduralpluralistic,
        title={MoReBench: Evaluating Procedural and Pluralistic Moral Reasoning in Language Models, More than Outcomes},
        author={Yu Ying Chiu and Michael S. Lee and Rachel Calcott and Brandon Handoko and Paul de Font-Reaulx and Paula Rodriguez and Chen Bo Calvin Zhang and Ziwen Han and Udari Madhushani Sehwag and Yash Maurya and Christina Q Knight and Harry R. Lloyd and Florence Bacus and Mantas Mazeika and Bing Liu and Yejin Choi and Mitchell L Gordon and Sydney Levine},
        year={2025},
        eprint={2510.16380},
        archivePrefix={arXiv},
        primaryClass={cs.CL},
        url={https://arxiv.org/abs/2510.16380},
  }



```
MIT License

Copyright (c) 2025 MoReBench

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
