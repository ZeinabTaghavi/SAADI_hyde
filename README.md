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

## Standalone HyDE on LooGLE

The repository also contains a standalone retrieval experiment for LooGLE. It does not import anything from the parent SAADI repository and can be copied to another server by itself.

The default run reproduces the population in the saved HippoRAG comparison:

- LooGLE `shortdep_qa`, test split
- the frozen set of 25 document IDs in `configs/loogle_hipporag_subset.json`
- 859 sentence-aware chunks of at most 500 whitespace-delimited words, without overlap
- 534 questions with chunk-level evidence labels
- per-document `facebook/contriever` retrieval
- top-5 and top-10 metrics and table rows

HyDE generation matches `hyde-dl19.ipynb`: local `Qwen/Qwen3-30B-A3B-Instruct-2507`, eight hypothetical passages, 512 new tokens, temperature 0.7, top-p 0.8, and `device_map="auto"`. The retrieval vector is the arithmetic mean of the normalized Contriever embeddings for the original question and all eight passages.

### Installation on the experiment server

Create an isolated environment and install a CUDA-compatible PyTorch build for that server. Then run:

```bash
pip install -r requirements-loogle.txt
pip install -e . --no-deps
```

The second command installs this local package without reinstalling PyTorch. The LooGLE runner uses Transformers directly and does not require Java, Pyserini, or the DL19 index.

Configure the model and dataset caches. The launcher uses the same defaults as the working notebook, but every value can be overridden:

```bash
export HF_TOKEN=<your-hugging-face-token>
export SAADI_HF_CACHE_ROOT=/mnt/cache/taghavi
export GPUS=4,5,6,7
```

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
```

Hypothetical documents are appended to `hyde_runs/loogle/hyde/<run-name>/hypotheses.jsonl` after every completed question. Document embeddings are cached beside them. If the process is interrupted, rerun the same command; resume is enabled by default. Use `--no-resume` to regenerate hypotheses, `--force-embeddings` to rebuild document embeddings, and `--force` to overwrite completed evaluation artifacts while retaining caches.

Useful overrides include:

```bash
GPUS=0,1,2,3 ./run_loogle_hyde.sh
./run_loogle_hyde.sh --embedding-device cuda:0
./run_loogle_hyde.sh --top-ks 5 10 --log-level INFO
```

### Evaluation artifacts and tables

Results are written to:

```text
hyde_evaluations/loogle/hyde/top_5/loogle_retrieval_ablation_hyde/
hyde_evaluations/loogle/hyde/top_10/loogle_retrieval_ablation_hyde/
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

The evaluation includes Gold, Silver-Loose, and Union-Loose Recall/MRR/nDCG, plus Gold Hit, Silver-Strict Hit, and Strict-Union Hit. Generate paper-ready JSONL, CSV, Markdown, and LaTeX rows with:

```bash
python generate_hyde_retriever_table.py
```

The files are written under `hyde_evaluations_Tables/`.

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
