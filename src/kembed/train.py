import dataclasses
import functools
import json
import logging
import os

import deepspeed
import torch
import torch.nn.functional as F
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

from kembed.arguments import DataArguments, EntityEmbeddingArguments, ModelArguments
from kembed.configuration import KPRConfigForBert, KPRConfigForXLMRoberta
from kembed.data import KPRTrainingCollator, KPRTrainingDataset
from kembed.modeling import KPRModelForBert, KPRModelForXLMRoberta
from kembed.utils import load_tsv_mapping

logger = logging.getLogger(__name__)

# _init_weights does not work properly with DeepSpeed ZeRO 3 when initializing the entity embeddings
transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights = lambda self, _: None
transformers.models.xlm_roberta.modeling_xlm_roberta.XLMRobertaPreTrainedModel._init_weights = lambda self, _: None

MODEL_CLASS_MAPPING = {
    "bert": KPRModelForBert,
    "xlm-roberta": KPRModelForXLMRoberta,
}

CONFIG_CLASS_MAPPING = {
    "bert": KPRConfigForBert,
    "xlm-roberta": KPRConfigForXLMRoberta,
}


def _init_weights(module: torch.nn.Module, initializer_range: float):
    if isinstance(module, torch.nn.Linear):
        module.weight.data.normal_(mean=0.0, std=initializer_range)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, torch.nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=initializer_range)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, torch.nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


def main(
    data_args: DataArguments,
    model_args: ModelArguments,
    entity_embedding_args: EntityEmbeddingArguments,
    training_args: TrainingArguments,
):
    logger.info(f"Data Arguments: {data_args}")
    logger.info(f"Model Arguments: {model_args}")
    logger.info(f"Entity Embedding Arguments: {entity_embedding_args}")
    logger.info(f"Training Arguments: {training_args}")

    arguments = {
        "data_args": dataclasses.asdict(data_args),
        "model_args": dataclasses.asdict(model_args),
        "entity_embedding_args": dataclasses.asdict(entity_embedding_args),
        "training_args": dataclasses.asdict(training_args),
    }
    os.makedirs(training_args.output_dir, exist_ok=True)
    with open(os.path.join(training_args.output_dir, "arguments.json"), "w") as f:
        json.dump(arguments, f, indent=2)

    set_seed(training_args.seed)

    if model_args.tokenizer_name_or_path is not None:
        tokenizer_name_or_path = model_args.tokenizer_name_or_path
    else:
        tokenizer_name_or_path = model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=False)

    if model_args.entity_fusion_method == "none":
        entity_vocab_size = 0
        entity_embedding_size = 0
    else:
        entity_vocab = load_tsv_mapping(entity_embedding_args.entity_vocab_file, int)
        entity_vocab_size = len(entity_vocab)

        entity_embeddings = torch.load(
            os.path.join(entity_embedding_args.entity_embedding_dir, "embeddings.pt"), weights_only=True
        )
        entity_embedding_size = entity_embeddings.size(1)

    if model_args.config_name_or_path is not None:
        config_name_or_path = model_args.config_name_or_path
    else:
        config_name_or_path = model_args.model_name_or_path

    base_config = AutoConfig.from_pretrained(config_name_or_path)
    config = CONFIG_CLASS_MAPPING[model_args.base_model_type](
        entity_vocab_size=entity_vocab_size,
        entity_embedding_size=entity_embedding_size,
        entity_fusion_method=model_args.entity_fusion_method,
        use_entity_position_embeddings=model_args.use_entity_position_embeddings,
        entity_fusion_activation=model_args.entity_fusion_activation,
        num_entity_fusion_attention_heads=model_args.num_entity_fusion_attention_heads,
        similarity_function=model_args.similarity_function,
        similarity_temperature=model_args.similarity_temperature,
        **base_config.to_dict(),
    )

    logger.info(f"Config: {config}")

    model = MODEL_CLASS_MAPPING[model_args.base_model_type].from_pretrained(
        model_args.model_name_or_path, config=config, torch_dtype=torch.float16
    )
    if model_args.freeze_base_model:
        for param in model.bert.parameters():
            param.requires_grad = False

    if model_args.entity_fusion_method != "none":
        embedding_entity_vocab = load_tsv_mapping(
            os.path.join(entity_embedding_args.entity_embedding_dir, "entity_vocab.tsv"), int
        )

        target_entity_embeddings = torch.zeros(len(entity_vocab), config.entity_embedding_size)
        for kb_id, index in entity_vocab.items():
            if index == 0:
                assert kb_id == "[PAD]"
                continue
            target_entity_embeddings[index] = entity_embeddings[embedding_entity_vocab[kb_id]]
        del entity_embeddings, embedding_entity_vocab

        with deepspeed.zero.GatheredParameters(
            model.parameters(), modifier_rank=0, enabled=is_deepspeed_zero3_enabled()
        ):
            target_entity_embeddings = F.normalize(target_entity_embeddings, p=2, dim=-1)
            entity_embedding_norm = torch.norm(model.bert.embeddings.word_embeddings.weight, dim=1).mean().item()
            target_entity_embeddings = target_entity_embeddings * entity_embedding_norm

            model.entity_fusion_layer.apply(
                functools.partial(_init_weights, initializer_range=config.initializer_range)
            )
            entity_embeddings_module = model.entity_fusion_layer.entity_embeddings
            entity_embeddings_module.embeddings.weight.data.copy_(target_entity_embeddings)

            model.entity_fusion_layer.noop_embeddings.data.normal_(mean=0.0, std=config.initializer_range)

    dataset = KPRTrainingDataset(tokenizer=tokenizer, args=data_args)
    collator = KPRTrainingCollator(
        tokenizer, query_max_len=data_args.query_max_len, passage_max_len=data_args.passage_max_len
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset, data_collator=collator)

    tokenizer.save_pretrained(training_args.output_dir)

    trainer.train()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    parser = HfArgumentParser((DataArguments, ModelArguments, EntityEmbeddingArguments, TrainingArguments))
    data_args, model_args, entity_embedding_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.local_rank == 0:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING

    logging.basicConfig(
        level=log_level, format="[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)"
    )
    transformers.utils.logging.set_verbosity(logging.WARNING)

    main(data_args, model_args, entity_embedding_args, training_args)
