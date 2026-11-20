# BeatLedger

BeatLedger turns timestamped microdrama screenplays into ranked 20–60 second promo cuts. It proposes one continuous excerpt or a two-part cut from consecutive episodes. Every edit point and cited beat can be traced to the screenplay.

## Try it

Open the [hosted Streamlit app](https://beatledger.streamlit.app/), or run the same app locally with Python 3.11 or newer.

```sh
python -m venv .venv
# Activate .venv for your shell, then run these commands from the repo root.
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.address 127.0.0.1
```

- **Bundled sample** runs three synthetic episodes through the full offline pipeline. The sample includes a stitch that skips a recap and a silent cut that requires footage confirmation.
- **Open saved model run** shows nine real local-model judgments on those same synthetic scripts. No API key is needed.
- **Upload files** accepts UTF-8 `.txt` screenplays. Use scripts you can share with the deployment. [Screenplay format example](tests/fixtures/episode-017.txt)
- In the hosted app, choose Groq, Google Gemini, OpenAI or OpenRouter under **AI judge**, save a provider key for the session, then click **Run AI analysis**. You can change the model while reusing the key. The provider receives excerpts from up to 12 cuts per run. [Provider settings](docs/DEPLOYMENT.md#use-your-own-api-key)
- Inspect a cut and open **Adjust cut**. Drag the two handles on the range slider, check the exact timecodes in the preview, then apply the trim. Stitched cuts use the same control for the outer edges or the join. A valid edit must keep its total duration within 20–60 seconds.
- Open **Cut metrics** for the ranked cut and its baseline comparison. The **Metrics** tab shows run coverage, model usage, timing, review holds, and score and rank changes. Download ranked CSV, JSON, run audit and review decisions. Review decisions live in the browser session.

## How cuts are selected

The parser gives each beat an exact start and end in centiseconds and retains its source lines. Search considers dialogue-safe boundaries, removes near-duplicate alternatives, checks adjacent episodes for repeated openers, and keeps up to three continuous options per episode and one stitch per adjacent pair.

The baseline uses screenplay cues for hook, cliffhanger, clarity, and drama. An optional judge scores eligible opening and ending evidence without choosing timestamps. Strict validation checks the evidence IDs and derives tension change from the two ratings. Scoring combines 35% hook, 30% cliffhanger, 25% clarity, and 10% drama, then applies timing deductions and at most +0.35 for escalating tension.

The model can misunderstand the story even when its beat ID is valid. A stitch goes on review hold if the baseline rates the ending at most 6 and the judge rates it at least 8. Each judged cut shows its baseline rank and score for comparison. Editors confirm silent cuts against footage, and a revised cut requires a fresh review. Screenplay clocks and narrative scores need footage review before posting.

## Batch and tests

```sh
python -m unittest discover -s tests
python run_batch.py --input tests/fixtures/stitches --output raw_outputs/demo
python verify_run.py --input tests/fixtures/stitches --run raw_outputs/demo --expected-episodes 2
```

The output path must be new. To judge locally, start a local-only Ollama server with `OLLAMA_NO_CLOUD=1`, pull `qwen2.5:1.5b`, and add `--judge` to the batch command. Set `BEATLEDGER_MODEL`, `BEATLEDGER_BASE_URL`, and optionally `BEATLEDGER_API_KEY` to swap compatible providers. Remote batch judging requires HTTPS and explicit `--allow-remote-judge`. [Deployment and provider settings](docs/DEPLOYMENT.md)

## Files by phase

| Phase | Files and purpose |
| --- | --- |
| 0. Design | `README.md` explains usage, approach, and the file layout. |
| 1. Source parsing | `beatledger/__init__.py` marks the Python package. `parser.py` reads exact screenplay beats and provenance. `inventory.py` loads sources and reports input coverage. `.gitignore` excludes inputs, results, secrets and caches. `test_parser.py`, `test_inventory.py`, and `fixtures/episode-017.txt` exercise the format and diagnostics. |
| 2. Candidate search | `verifier.py` enforces source-backed cut and evidence boundaries. `recap.py` identifies replayed openers. `search.py` shortlists continuous and stitched cuts. `ranking.py` computes baseline scores and pacing. `run_batch.py` runs the shared pipeline and writes CSV, JSON, telemetry, and flags. `test_candidates.py` and `test_batch.py` verify these paths. `fixtures/stitches/episode-101.txt` and `episode-102.txt` form a synthetic replay and continuation. |
| 3. Judge | `judge.py` defines the rubric, response contract, and evidence checks. `judge_provider.py` supplies model transport and usage tracking. `requirements.txt` declares Pydantic and Streamlit. `test_judge.py` covers schema, evidence, ranking holds, retries, and fallback. |
| 4. Workbench | `app.py` supplies the editor UI. `beatledger/workbench.py` applies filters, verification and review rules. `.streamlit/config.toml` controls the theme and telemetry. `fixtures/workbench/episode-103.txt` adds a silent synthetic episode. `examples/judged_sample.json` stores genuine judgments on the bundled synthetic cuts. `test_workbench.py` and `test_workbench_app.py` exercise the editor workflow. |
| 5. Release | `verify_run.py` independently rechecks cut exports against the source files. `docs/DEPLOYMENT.md` explains public deployment and the hosted judge fallback. |

Fixture paths are relative to `tests/`. The initial assignment inputs and its batch results are shared privately with the evaluator.
