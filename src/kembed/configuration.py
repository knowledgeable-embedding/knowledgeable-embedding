from transformers.models.bert import BertConfig
from transformers.models.xlm_roberta import XLMRobertaConfig


def _init_function(
    self,
    entity_vocab_size: int | None = 10000,
    entity_embedding_size: int = 768,
    entity_fusion_method: str = "multihead_attention",
    use_entity_position_embeddings: bool = True,
    entity_fusion_activation: str = "softmax",
    num_entity_fusion_attention_heads: int = 12,
    similarity_function: str = "dot",
    similarity_temperature: float = 1.0,
    *args,
    **kwargs,
):
    self.entity_vocab_size = entity_vocab_size
    self.entity_embedding_size = entity_embedding_size
    self.entity_fusion_method = entity_fusion_method
    self.use_entity_position_embeddings = use_entity_position_embeddings
    self.entity_fusion_activation = entity_fusion_activation
    self.num_entity_fusion_attention_heads = num_entity_fusion_attention_heads
    self.similarity_function = similarity_function
    self.similarity_temperature = similarity_temperature

    super(self.__class__, self).__init__(*args, **kwargs)


class KPRConfigForBert(BertConfig):
    __init__ = _init_function
    model_type = "kpr-bert"


class KPRConfigForXLMRoberta(XLMRobertaConfig):
    __init__ = _init_function
    model_type = "kpr-xlm-roberta"
