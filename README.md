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

Although RAG typically relies on embedding-based retrieval, the embedding models themselves are also based on language models and therefore struggle with queries involving less-frequent entities ([Sciavolino et al., 2021](https://arxiv.org/abs/2109.08535)), often failing to retrieve the crucial knowledge needed to overcome this limitation.

**Knowledgeable Embedding** addresses this challenge by injecting real-world entity knowledge into embeddings, making them more *knowledgeable*.

**The entity knowledge is pluggable and can be dynamically updated.** See the [How to Update Entity Knowledge](#how-to-update-entity-knowledge) section for details.

## Models

| Model | Model Size | Base Model |
| --- | --- | --- |
| [knowledgeable-ai/kpr-bert-base-uncased](https://huggingface.co/knowledgeable-ai/kpr-bert-base-uncased) | 112M | [bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) |
| [knowledgeable-ai/kpr-retromae](https://huggingface.co/knowledgeable-ai/kpr-retromae) | 112M | [RetroMAE](https://huggingface.co/Shitao/RetroMAE) |
| [knowledgeable-ai/kpr-bge-base-en](https://huggingface.co/knowledgeable-ai/kpr-bge-base-en) | 112M | [bge-base-en](https://huggingface.co/BAAI/bge-base-en) |
| [knowledgeable-ai/kpr-bge-base-en-v1.5](https://huggingface.co/knowledgeable-ai/kpr-bge-base-en-v1.5) | 112M | [bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) |
| [knowledgeable-ai/kpr-bge-large-en-v1.5](https://huggingface.co/knowledgeable-ai/kpr-bge-large-en-v1.5) | 340M | [bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) |

For practical use, we recommend `knowledgeable-ai/kpr-bge-*`, which significantly outperforms state-of-the-art models on queries involving less-frequent entities while performing comparably on other queries, as reported in [our paper](https://arxiv.org/abs/2507.03922).

Regarding the model size, we do not count the entity embeddings since they are stored in CPU memory and have a negligible impact on runtime performance. See [this page](https://github.com/knowledgeable-embedding/knowledgeable-embedding/wiki/Internals-of-Knowledgeable-Embedding) for details.

## Usage

Our models can be used via [Hugging Face Transformers](https://github.com/huggingface/transformers) or [Sentence Transformers](https://github.com/UKPLab/sentence-transformers):

### Hugging Face Transformers

```python
from transformers import AutoTokenizer, AutoModel
import torch

MODEL_NAME_OR_PATH = "knowledgeable-ai/kpr-bge-base-en"

input_texts = [
  "Who founded Dominican Liberation Party?",
  "Who owns Mompesson House?"
]

# Load model and tokenizer from the Hugging Face Hub
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_NAME_OR_PATH, trust_remote_code=True)

# Preprocess the text
preprocessed_inputs = tokenizer(input_texts, return_tensors="pt", padding=True)

# Compute embeddings
with torch.no_grad():
  embeddings = model.encode(**preprocessed_inputs)

print("Embeddings:", embeddings)
```

### Sentence Transformers

```python
from sentence_transformers import SentenceTransformer

MODEL_NAME_OR_PATH = "knowledgeable-ai/kpr-bge-base-en"

input_texts = [
  "Who founded Dominican Liberation Party?",
  "Who owns Mompesson House?"
]

# Load model from the Hugging Face Hub
model = SentenceTransformer(MODEL_NAME_OR_PATH, trust_remote_code=True)

# Compute embeddings
embeddings = model.encode(input_texts)

print("Embeddings:", embeddings)
```

**IMPORTANT:** This code will be supported in versions of Sentence Transformers later than v5.1.0, which have not yet been released at the time of writing. Until then, please install the library directly from GitHub:

```bash
pip install git+https://github.com/UKPLab/sentence-transformers.git
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

The entity knowledge injected into embeddings can be easily updated without retraining. The knowledge is stored in dedicated entity embeddings included in the Hugging Face-compatible tokenizer. The tokenizer uses an entity linker to detect entities mentioned in the text and outputs their corresponding embeddings. To update the knowledge, you need to build a model together with a new tokenizer that includes updated entity embeddings and an updated entity linker. Refer to [this wiki page](https://github.com/knowledgeable-embedding/knowledgeable-embedding/wiki/Internals-of-Knowledgeable-Embedding) for technical details.

### Updating Entity Knowledge from Wikipedia

The entity embeddings and entity linker can be built by following the instructions on the  
[Building Entity Linker and Entity Embeddings from Wikipedia](https://github.com/knowledgeable-embedding/knowledgeable-embedding/wiki/Building-Entity-Linker-and-Entity-Embeddings-from-Wikipedia) wiki page.

To update entity knowledge, create a new model by running `build_hf_model.py` with the updated entity vocabulary, embeddings, and linker.  
You can then push the model to the Hugging Face Hub by specifying the repository ID with `--hf_repo_id`.  
Adding the `--private` flag will upload the model as private.

```bash
# Target model for updating entity knowledge
MODEL_NAME_OR_PATH = "knowledgeable-ai/kpr-bge-base-en"

python scripts/build_hf_model.py \
  --model_name_or_path ${MODEL_NAME_OR_PATH} \
  --entity_linker_dir <ENTITY_LINKER_DIR> \
  --entity_embedding_dir <ENTITY_EMBEDDING_DIR> \
  --entity_vocab_file <ENTITY_VOCAB_FILE> \
  --output_dir <OUTPUT_DIR> \
  --hf_repo_id your_name/tokenizer_name
```

The model can be loaded locally:

```python
# Hugging Face Transformers
model = AutoModel.from_pretrained("<OUTPUT_DIR>", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("<OUTPUT_DIR>", trust_remote_code=True)

# Sentence Transformers
model = SentenceTransformer("<OUTPUT_DIR>", trust_remote_code=True)
```

Or directly from the Hugging Face Hub (if uploaded):

```python
# Hugging Face Transformers
model = AutoModel.from_pretrained("your_name/model_name", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("your_name/model_name", trust_remote_code=True)

# Sentence Transformers
model = SentenceTransformer("your_name/model_name", trust_remote_code=True)
```

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
