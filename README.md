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

The notebooks load Qwen locally with Transformers using `device_map="auto"`, so `CUDA_VISIBLE_DEVICES` controls which GPUs are used and the `HF_HOME`/`TRANSFORMERS_CACHE` settings control where model files are cached.


## Citation

```
@article{hyde,
  title = {Precise Zero-Shot Dense Retrieval without Relevance Labels},
  author = {Luyu Gao and Xueguang Ma and Jimmy Lin and Jamie Callan},
  journal={arXiv preprint arXiv:2212.10496},
  year = {2022}
}
```
