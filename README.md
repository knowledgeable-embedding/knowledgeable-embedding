<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="resources/logo-dark.png">
    <source media="(prefers-color-scheme: light)" srcset="resources/logo-light.png">
    <img alt="Knowledgeable Embedding" src="resources/logo-light.png" width="400" height="140" style="max-width: 100%;">
  </picture>
</p>
<h3 align="center">
    <p>Injecting dynamically updatable entity knowledge into embeddings to enhance RAG</p>
</h3>
<br />

A key limitation of large language models (LLMs) is their inability to capture less-frequent or up-to-date entity knowledge, often leading to factual inaccuracies and hallucinations. Retrieval-augmented generation (RAG), which incorporates external knowledge through retrieval, is a common approach to mitigate this issue.

Although RAG typically relies on embedding-based retrieval, the embedding models themselves are also based on language models and therefore struggle with queries involving less-frequent entities, often failing to retrieve the crucial knowledge needed to overcome this limitation.

**Knowledgeable Embedding** is a toolkit that addresses this challenge by injecting real-world entity knowledge into embeddings, making them more *knowledgeable*. **The entity knowledge is pluggable and can be dynamically updated with ease**.

## Models

| Model | Model Size | Base Model |
| --- | --- | --- |
| [knowledgeable-ai/kpr-bert-base-uncased](https://huggingface.co/knowledgeable-ai/kpr-bert-base-uncased) | 112M | [bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) |
| [knowledgeable-ai/kpr-retromae](https://huggingface.co/knowledgeable-ai/kpr-retromae) | 112M | [RetroMAE](https://huggingface.co/Shitao/RetroMAE) |
| [knowledgeable-ai/kpr-bge-base-en](https://huggingface.co/knowledgeable-ai/kpr-bge-base-en) | 112M | [bge-base-en](https://huggingface.co/BAAI/bge-base-en) |
| [knowledgeable-ai/kpr-bge-base-en-v1.5](https://huggingface.co/knowledgeable-ai/kpr-bge-base-en-v1.5) | 112M | [bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) |
| [knowledgeable-ai/kpr-bge-large-en-v1.5](https://huggingface.co/knowledgeable-ai/kpr-bge-large-en-v1.5) | 340M | [bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) |

For practical use, we recommend `knowledgeable-ai/kpr-bge-*`, which significantly outperforms other models on queries involving less-frequent entities while performing comparably on other queries, as reported in [our paper](https://arxiv.org/abs/2507.03922).

Regarding the model size, we do not count the entity embeddings since they are stored in CPU memory and have a negligible impact on runtime performance. See [this page](https://github.com/knowledgeable-embedding/knowledgeable-embedding/wiki/Internals-of-Knowledgeable-Embedding) for details.

## Usage

```python
from transformers import AutoTokenizer, AutoModel
import torch

MODEL_NAME_OR_PATH = "knowledgeable-ai/kpr-bge-base-en"

input_texts = [
  "Who founded Dominican Liberation Party?",
  "Who owns Mompesson House?"
]

# Load model and tokenizer from Hugging Face Hub
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_NAME_OR_PATH, trust_remote_code=True)

# Preprocess the text
preprocessed_inputs = tokenizer(input_texts, return_tensors="pt", padding=True)

with torch.no_grad():
    embeddings = model.encode(**preprocessed_inputs)

print("Embeddings:", embeddings)
```

## Installation

After cloning this repository, run the following command to install the toolkit:

```bash
pip install -r requirements.txt
pip install .
```

## What is Entity?

In this toolkit, an entity refers to an entry in a knowledge base (i.e., a set of entities). While Wikipedia is used as the default knowledge base, any other knowledge base can be used as the source of entity knowledge. In the case of Wikipedia, each entry (e.g., a person, organization, or technical term) is treated as an entity.

## How to Update Entity Knowledge

The entity knowledge injected into embeddings can be easily updated without retraining. The knowledge is stored in dedicated entity embeddings included in the Hugging Face-compatible tokenizer. The tokenizer uses an entity linker to detect entities mentioned in the text and outputs their corresponding embeddings. To update the knowledge, a new tokenizer must be built with updated entity embeddings and an updated entity linker. Refer to [this wiki page](https://github.com/knowledgeable-embedding/knowledgeable-embedding/wiki/Internals-of-Knowledgeable-Embedding) for technical details.

### Updating Entity Knowledge from Wikipedia

The entity embeddings and entity linker can be easily built by following the instructions on the [Building Entity Linker and Entity Embeddings from Wikipedia](https://github.com/knowledgeable-embedding/knowledgeable-embedding/wiki/Building-Entity-Linker-and-Entity-Embeddings-from-Wikipedia) wiki page.

A new tokenizer with updated entity knowledge can be created by running `build_hf_tokenizer.py` with the updated entity vocabulary, entity embeddings, and entity linker.
The tokenizer can be pushed to the Hugging Face Hub by specifying the repository ID with `--hf_repo_id`.

```bash
python scripts/build_hf_tokenizer.py \
  --output_dir <OUTPUT_DIR> \
  --entity_linker_dir <ENTITY_LINKER_DIR> \
  --entity_embedding_dir <ENTITY_EMBEDDING_DIR> \
  --entity_vocab_file <ENTITY_VOCAB_FILE> \
  --hf_repo_id your_name/tokenizer_name
```

The tokenizer can also be loaded locally:

```python
>>> tokenizer = AutoTokenizer.from_pretrained("<OUTPUT_DIR>", trust_remote_code=True)
```

Or directly from the Hugging Face Hub (if uploaded):

```python
>>> tokenizer = AutoTokenizer.from_pretrained("your_name/tokenizer_name", trust_remote_code=True)
```

New entity knowledge can be injected by using a new tokenizer instead of the tokenizer bundled with the model.

### Updating Entity Knowledge from Custom Data Source

(TBD)

## Internals of Knowledgeable Embedding

Technical details of this toolkit are provided on the [Internals of Knowledgeable Embedding](https://github.com/knowledgeable-embedding/knowledgeable-embedding/wiki/Internals-of-Knowledgeable-Embedding) page.

## Training from Scratch

Training and evaluation can be performed by following the instructions on the [Training and Evaluation](https://github.com/knowledgeable-embedding/knowledgeable-embedding/wiki/Training-and-Evaluation) page.

## Citation

If you use this code or models in your research, please cite the following paper:

[Dynamic Injection of Entity Knowledge into Dense Retrievers](https://arxiv.org/abs/2507.03922)

```bibtex
@article{yamada2025kpr,
  title={Dynamic Injection of Entity Knowledge into Dense Retrievers},
  author={Ikuya Yamada and Ryokan Ri and Takeshi Kojima and Yusuke Iwasawa and Yutaka Matsuo},
  journal={arXiv preprint arXiv:2507.03922},
  year={2025}
}
```
