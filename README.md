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

**Knowledgeable Embedding** is a toolkit that addresses this challenge by injecting real-world entity knowledge into embeddings, making them more *knowledgeable*. The entity knowledge is pluggable and can be dynamically updated with ease.

## Models

(TBD)

## Usage

(TBD)

## Installation

After cloning this repository, run the following command to install the toolkit:

```bash
pip install -r requirements.txt
pip install .
```

## What is Entity?

In this toolkit, an entity refers to an entry in a knowledge base (i.e., a set of entities). While Wikipedia is used as the default knowledge base, any other knowledge base can be used as the source of entity knowledge. In the case of Wikipedia, each entry (e.g., a person, organization, or technical term) is treated as an entity.

## How to Update Entity Knowledge

Since our model learns to attend to entity knowledge relevant to the input text, the knowledge can be easily updated without retraining. It is stored in dedicated entity embeddings included in the Hugging Face-compatible tokenizer. The tokenizer uses an entity linker to detect entities mentioned in the text and outputs their corresponding embeddings. To update the knowledge, a new tokenizer must be built with updated entity embeddings and an updated entity linker. Refer to [the section below](#internals-of-knowledgeable-embedding) for details.

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

### Updating Entity Knowledge from Custom Data Source

(TBD)

## Internals of Knowledgeable Embedding

This toolkit is currently based on the Knowledgeable Passage Retriever (KPR) model, a BERT-based retriever enhanced with a context-entity attention layer and dynamically updatable entity embeddings, as described in [our paper](#citation).  

We highlight four key components of this toolkit: the entity linker, entity embeddings, the knowledgeable tokenizer, and the context-entity attention layer.

### Entity Linker

The entity linker detects entities mentioned in the input text using dictionary-based string matching. The dictionary consists of entity names (e.g., "Apple") and their possible referent entities (e.g., *Apple Inc.* and *Apple (food)*). We tokenize text using the default English tokenizer provided by the SpaCy library and extract entity names by matching n-grams against the dictionary. For ambiguous names, we do not disambiguate to a single entity but instead include all possible candidates.

The entity linker is built directly from Wikipedia. It is lightweight because it only involves tokenization, string matching with an efficient [trie algorithm](https://en.wikipedia.org/wiki/Trie), and dictionary lookups.

### Entity Embeddings

The entity knowledge injected into embeddings is stored in dedicated entity embeddings. These embeddings are computed using single-pass BERT inference over Wikipedia passages that refer to the corresponding entity. Since they are kept frozen during training, they can be updated without retraining.

The total size of these embeddings is typically large. For example, embeddings based on a base-sized BERT model derived from the entire English Wikipedia contain approximately 7.2 million 768-dimensional vectors, each representing a Wikipedia entity (7.2M × 768 = 5.5 billion parameters, about 11 GB in float16).  
For efficiency, they are stored in CPU memory as a memory-mapped NumPy array. This design allows the model to run on consumer-grade GPUs with limited memory.

### Knowledgeable Tokenizer

The Knowledgeable Tokenizer is a Hugging Face-compatible tokenizer that includes an entity linker and entity embeddings. It automatically detects entities mentioned in the input text and outputs their corresponding entity embeddings along with the usual tokenization data.  

The entity knowledge injected into embeddings can be updated by building a new tokenizer with updated entity embeddings and an entity linker, as described in the [How to Update Entity Knowledge](#how-to-update-entity-knowledge) section.

### Context-entity Attention Layer

The KPR model extends a BERT-based retriever with a context-entity attention layer placed on top. This attention layer attends to entity embeddings relevant to the input text. We find that this design makes the model robust to noise from the entity linker, such as the incorrect detection of **Apple (food)** for the mention "Apple" in computer-related texts.

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
