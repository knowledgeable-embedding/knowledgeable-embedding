from dataclasses import dataclass, field


@dataclass
class DataArguments:
    training_dataset_dir: str = field(metadata={"help": "Path or glob pattern for the training dataset directory"})
    query_dataset_dir: str = field(metadata={"help": "Path or glob pattern for the query dataset directory"})
    passage_dataset_dir: str = field(metadata={"help": "Path or glob pattern for the passage dataset directory"})

    sample_negatives_from_topk: int = field(
        default=10000,
        metadata={"help": "Number of top-ranked passages to sample negatives from"},
    )
    train_group_size: int = field(
        default=2,
        metadata={"help": "Number of passages per query (1 positive + N negatives)"},
    )

    query_max_len: int = field(
        default=32,
        metadata={"help": "Maximum token length for queries"},
    )
    passage_max_len: int = field(
        default=256,
        metadata={"help": "Maximum token length for passages"},
    )
    query_prefix: str = field(
        default="",
        metadata={"help": "Optional prefix to add to each query"},
    )
    passage_prefix: str = field(
        default="",
        metadata={"help": "Optional prefix to add to each passage"},
    )


@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Path or name of the base model"})
    tokenizer_name_or_path: str | None = field(
        default=None,
        metadata={"help": "Path or name of the tokenizer (defaults to model_name_or_path if None)"},
    )
    config_name_or_path: str | None = field(
        default=None,
        metadata={"help": "Path or name of the model config (defaults to model_name_or_path if None)"},
    )

    base_model_type: str = field(
        default="bert", metadata={"help": "Base model type", "choices": ["bert", "xlm-roberta"]}
    )
    entity_fusion_method: str = field(
        default="multihead_attention",
        metadata={
            "help": "Method to fuse entity embeddings with the resulting embedding",
            "choices": ["multihead_attention", "none"],
        },
    )
    use_entity_position_embeddings: bool = field(
        default=True,
        metadata={"help": "Whether to use entity position embeddings"},
    )
    entity_fusion_activation: str = field(
        default="sigmoid",
        metadata={"help": "Activation function for entity fusion", "choices": ["sigmoid", "softmax"]},
    )
    num_entity_fusion_attention_heads: int = field(
        default=1,
        metadata={"help": "Number of attention heads for entity fusion"},
    )
    similarity_function: str = field(
        default="dot",
        metadata={"help": "Similarity function for retrieval", "choices": ["dot", "cosine"]},
    )
    similarity_temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature parameter for similarity activation"},
    )
    freeze_base_model: bool = field(
        default=False,
        metadata={"help": "Whether to freeze the base model during training"},
    )


@dataclass
class EntityEmbeddingArguments:
    entity_vocab_file: str | None = field(
        default=None,
        metadata={"help": "Path to the entity vocabulary TSV file"},
    )
    entity_embedding_dir: str | None = field(
        default=None,
        metadata={"help": "Path to the directory containing entity embeddings"},
    )
