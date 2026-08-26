# HyDE: Precise Zero-Shot Dense Retrieval without Relevance Labels

This is code repository for the paper: [HyDE: Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496).

**HyDE** zero-shot instructs GPT3 to generate a fictional document and re-encodes it with unsupervised retriever Contriever to search in its embedding space.
HyDE significantly outperforms Contriever across tasks and languages and it does not require any human labeled relevance judgement.

![approach](approach.png)

## Steps to run the code

1. Install this package and its Python dependencies.

```
pip install -e .
```

2. Install `pyserini` by following the [guide](https://github.com/castorini/pyserini#-installation). We use pyserini to conduct dense retrieval and evaluation.


3. Download the prebuilt Contrever faiss index
```
wget  https://www.dropbox.com/s/dytqaqngaupp884/contriever_msmarco_index.tar.gz
tar -xvf contriever_msmarco_index.tar.gz
```

4. Setup Hugging Face API token for Qwen generation

```
export HF_TOKEN=<your Hugging Face token>
```

5. Run `hyde-dl19.ipynb`, it will run the experiment on the TREC DL19 dataset with `Qwen/Qwen3-30B-A3B-Instruct-2507`. Run `hyde-demo.ipynb`, it will go through HyDE pipeline with an example query.

The notebooks load Qwen locally with Transformers using `device_map="auto"`, so `CUDA_VISIBLE_DEVICES` controls which GPUs are used and `HF_HUB_CACHE` controls where the sharded Qwen model snapshot is cached.

## Standalone HyDE benchmark runners

The repository also contains standalone retrieval experiments for LooGLE,
QASPER-64K, MuSiQue-32K, NovelHopQA, and the legacy unexpanded QASPER subset.
They do not import anything from the parent SAADI repository and can be copied
to another server by themselves.

The default run reproduces the population in the saved HippoRAG comparison:

- LooGLE `shortdep_qa`, test split
- the frozen set of 25 document IDs in `configs/loogle_hipporag_subset.json`
- 859 sentence-aware chunks of at most 500 whitespace-delimited words, without overlap
- 534 questions with chunk-level evidence labels
- per-document `facebook/contriever` retrieval
- top-5 and top-10 metrics and table rows

HyDE generation matches `hyde-dl19.ipynb`: local `Qwen/Qwen3-30B-A3B-Instruct-2507`, eight hypothetical passages, 512 new tokens, temperature 0.7, top-p 0.8, and `device_map="auto"`. The retrieval vector is the arithmetic mean of the normalized Contriever embeddings for the original question and all eight passages.

### Installation on the experiment server

Create an isolated environment in this standalone folder and install a
CUDA-compatible PyTorch build for that server. The launchers prefer
`.venv/bin/python`, so installation and execution cannot accidentally use
different Python environments:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-loogle.txt
python -m pip install -e . --no-deps
```

The second command installs this local package without reinstalling PyTorch. The LooGLE runner uses Transformers directly and does not require Java, Pyserini, or the DL19 index.

Configure the model and dataset caches. The launcher uses the same defaults as the working notebook, but every value can be overridden:

```bash
export HF_TOKEN=<your-hugging-face-token>
export SAADI_HF_CACHE_ROOT=/mnt/cache/taghavi
```

All launchers default to physical GPUs `0,1,2,3`, set both
`CUDA_VISIBLE_DEVICES` and `GLOBAL_VISIBLE_DEVICES`, and verify that PyTorch can
see all four GPUs before loading a model. Set `GPUS` only when an intentional
override is needed. `HF_HOME`, `HF_HUB_CACHE`, `HF_DATASETS_CACHE`, and
`TRANSFORMERS_CACHE` are derived from `SAADI_HF_CACHE_ROOT`.

If every model and dataset file is already cached, offline mode is supported:

```bash
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

### Validation, smoke test, and full run

First verify the frozen population without loading Qwen or Contriever:

```bash
./run_loogle_hyde.sh --validate-only
```

This must report 25 documents, 859 chunks, 534 retrieval examples, and average chunk size `479.371362048894`. A mismatch stops the run before either model is loaded.

Validate the other frozen HippoRAG comparison populations with:

```bash
./run_qasper_hyde.sh --validate-only

export NOVELHOPQA_BOOKS_ROOT=/path/to/novelhopqa/book-corpus-root
./run_novelhopqa_hyde.sh --validate-only
```

QASPER must report 25 documents, 880 chunks, and 80 labeled queries with the
canonical 100-word chunk size. NovelHopQA must report 18 books, 7,736 chunks,
and 985 labeled queries. NovelHopQA requires the external whole-book corpus
containing `bookmeta.json` and `Books/`; the launcher automatically detects the
parent repository corpus when it is available.

Run a small real-model smoke test:

```bash
./run_loogle_hyde.sh \
  --max-documents 1 \
  --max-qa-entries 10 \
  --run-name loogle_hyde_smoke
```

Run the full HippoRAG-comparable experiment:

```bash
./run_loogle_hyde.sh
./run_qasper_hyde.sh
./run_novelhopqa_hyde.sh
```

Run the expanded QASPER-64K and MuSiQue-32K matrix with SAADI's canonical
100-word, zero-overlap chunking. `--force` replaces artifacts created with the
old chunk configuration:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
./run_qasper64k_musique32k_hyde_gpu0_3.sh --force
```

To run only MuSiQue-32K:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
./run_musique32k_hyde_saadi_chunk100_gpu0_3.sh
```

This launcher reuses compatible HyDE hypotheses and force-regenerates the five
retrievers' top-5/top-10 artifacts with 100-word chunks.

Hypothetical documents are appended to `hyde_runs/<dataset>/hyde/<run-name>/hypotheses.jsonl` after every completed question. Document embeddings are cached beside them. If the process is interrupted, rerun the same command; resume is enabled by default. Use `--no-resume` to regenerate hypotheses, `--force-embeddings` to rebuild document embeddings, and `--force` to overwrite completed evaluation artifacts while retaining caches.

### LooGLE + NovelHopQA retriever matrix

The standalone matrix supports BM25, Contriever, BGE-M3, Qwen embeddings, and
Jina. E5 is intentionally excluded. Hypothetical passages are generated once
per dataset and reused unchanged by every retriever.

Check what has already completed:

```bash
./run_all_hyde_retrievers.sh --check-only
```

Run every missing experiment and regenerate the main-style table:

```bash
./run_all_hyde_retrievers.sh
```

If the copied environment has not been populated yet, the matrix launcher can
install into the exact Python interpreter that it will subsequently use:

```bash
bash ./run_all_hyde_retrievers.sh --install-deps
```

Before loading the dataset or a model, the launcher checks all required Python
packages, cache writability, and GPU visibility. Missing packages are reported
with an interpreter-specific repair command.

Useful filters and controls:

```bash
DATASETS_CSV=loogle RETRIEVERS_CSV=bm25,contriever ./run_all_hyde_retrievers.sh
GPUS=0,1,2,3 HYDE_QWEN_DEVICE_MAP=auto ./run_all_hyde_retrievers.sh
./run_all_hyde_retrievers.sh --dry-run
```

### QASPER-64K + MuSiQue-32K main-table matrix

The exact expanded datasets used by the main SAADI table are bundled under
`data/qasper_64k/` and `data/musique_32k/`. Every source file is validated
against its frozen SHA-256 manifest before a run starts:

- QASPER-64K: 23 expanded groups, 1,372 questions, 1,353 questions with
  retrieval evidence
- MuSiQue-32K: 45 expanded groups, 900 questions, with 300 questions from each
  of the 2-hop, 3-hop, and 4-hop populations

Run all five retrievers on physical GPUs 0, 1, 2, and 3 with:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash run_qasper64k_musique32k_hyde_gpu0_3.sh
```

The launcher first resumes the shared Qwen HyDE hypothesis cache for each
dataset using all four GPUs. It then runs BM25, Contriever, BGE-M3, Jina, and
Qwen retrieval in four independent GPU queues. Completed artifacts are skipped
only after their manifests, configurations, populations, and top-5/top-10
outputs validate. Useful controls are:

```bash
bash run_qasper64k_musique32k_hyde_gpu0_3.sh --check-only
bash run_qasper64k_musique32k_hyde_gpu0_3.sh --dry-run
bash run_qasper64k_musique32k_hyde_gpu0_3.sh --force
```

The Qwen3-MoE generator requires `transformers>=4.51.0`. The runtime preflight
checks architecture support in the selected interpreter before loading a model.
If a copied folder inherits an older unrelated virtual environment, repair it
and continue the resumable run with:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash run_qasper64k_musique32k_hyde_gpu0_3.sh --install-deps
```

After all workers succeed, strict table generation requires exactly 20 rows:
four finalized datasets times five retrievers. The legacy unexpanded QASPER
subset is intentionally excluded. The final JSONL, CSV, Markdown, LaTeX, and
text tables are written under `hyde_evaluations_Tables/`.

Both matrix launchers first run a generation-only process for each dataset.
They then run each retriever in a separate process with `--retrieval-only`, preventing
the 30B generation model and dense retriever models from remaining loaded
together. BM25 combines the original question and hypothetical passages by
averaging their per-document min-max-normalized score vectors. Dense retrievers
average their normalized embeddings.

Useful overrides include:

```bash
GPUS=0,1,2,3 ./run_loogle_hyde.sh
./run_loogle_hyde.sh --embedding-device cuda:0
./run_loogle_hyde.sh --top-ks 5 10 --log-level INFO
```

### Evaluation artifacts and tables

Results are written to:

```text
hyde_evaluations/loogle/hyde/<retriever>/top_5/loogle_retrieval_ablation_hyde/
hyde_evaluations/loogle/hyde/<retriever>/top_10/loogle_retrieval_ablation_hyde/
hyde_evaluations/qasper_64k/hyde/<retriever>/top_10/qasper_64k_retrieval_ablation_hyde/
hyde_evaluations/musique_32k/hyde/<retriever>/top_10/musique_32k_retrieval_ablation_hyde/
hyde_evaluations/novelhopqa/hyde/<retriever>/top_10/novelhopqa_retrieval_ablation_hyde/
```

Each directory contains:

```text
index/chunk_index.jsonl
index/index_stats.json
retrieval/retrieval_examples.jsonl
retrieval/retrieval_payloads.jsonl
retrieval/retrieval_results.json
metrics_per_query.jsonl
metrics_summary.json
leaderboard_row.json
evaluation_manifest.json
```

The evaluation includes Gold, Silver-Loose, and Union-Loose Recall/MRR/nDCG,
plus Gold Hit, Silver-Strict Hit, and Strict-Union Hit. Generate a table with
the exact columns of the project’s main retrieval table with:

```bash
./generate_results_table.sh
cat hyde_evaluations_Tables/table_main_retrieval_hyde.txt
```

The generator merges each dataset/retriever pair’s `top_5` and `top_10`
outputs into one row labeled `Method = HyDE`. Its ranking and binary metric
columns match the main SAADI retrieval table. JSONL, CSV, Markdown,
LaTeX, and `.txt` files are written under `hyde_evaluations_Tables/`.

### Tests

The local test suite uses mocked model components and does not download Qwen, Contriever, or LooGLE:

```bash
PYTHONPATH=src pytest -q
```


## Citation

```
@article{hyde,
  title = {Precise Zero-Shot Dense Retrieval without Relevance Labels},
  author = {Luyu Gao and Xueguang Ma and Jimmy Lin and Jamie Callan},
  journal={arXiv preprint arXiv:2212.10496},
  year = {2022}
}
```
