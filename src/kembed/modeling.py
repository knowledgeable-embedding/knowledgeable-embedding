import math

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import PretrainedConfig
from transformers.file_utils import ModelOutput
from transformers.models.bert import BertModel, BertPreTrainedModel
from transformers.models.xlm_roberta import XLMRobertaModel, XLMRobertaPreTrainedModel

from .configuration import KPRConfigForBert, KPRConfigForXLMRoberta


class EntityEmbeddings(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config

        if config.entity_vocab_size is not None:
            self.embeddings = nn.Embedding(config.entity_vocab_size, config.entity_embedding_size, padding_idx=0)
            self.embeddings.weight.requires_grad = False

        # The 0-th position corresponds to the [CLS] token which does not correspond to any entity
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size, padding_idx=0)

        self.dense = nn.Linear(config.entity_embedding_size, config.hidden_size, bias=False)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, entity_ids: Tensor | None, entity_embeds: Tensor | None, entity_position_ids: Tensor) -> Tensor:
        if entity_embeds is not None:
            entity_embeddings = entity_embeds
        elif entity_ids is not None:
            if self.config.entity_vocab_size is None:
                raise ValueError("Entity embeddings are not constructed because entity_vocab_size is None.")
            entity_embeddings = self.embeddings(entity_ids)
        else:
            raise ValueError("Either entity_ids or entity_embeds need to be provided.")

        entity_embeddings = self.dense(entity_embeddings)

        if self.config.use_entity_position_embeddings:
            entity_position_embeddings = self.position_embeddings(
                entity_position_ids
            )  # batch, entities, positions, hidden
            entity_position_embeddings = torch.sum(entity_position_embeddings, dim=2)
            entity_position_embeddings = entity_position_embeddings / entity_position_ids.ne(0).sum(dim=2).clamp(
                min=1
            ).unsqueeze(-1)
            entity_embeddings = entity_embeddings + entity_position_embeddings

        entity_embeddings = self.LayerNorm(entity_embeddings)
        entity_embeddings = self.dropout(entity_embeddings)

        return entity_embeddings


class EntityFusionMultiHeadAttention(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config

        self.num_attention_heads = config.num_entity_fusion_attention_heads
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)

        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, key_padding_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_layer = self.transpose_for_scores(self.query(query))
        key_layer = self.transpose_for_scores(self.key(key))
        value_layer = self.transpose_for_scores(self.value(value))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        dtype = attention_scores.dtype
        key_padding_mask_scores = key_padding_mask[:, None, None, :]
        key_padding_mask_scores = key_padding_mask_scores.to(dtype=dtype)
        key_padding_mask_scores = key_padding_mask_scores * torch.finfo(dtype).min
        attention_scores = attention_scores + key_padding_mask_scores
        orig_attention_scores = attention_scores.clone()

        if self.config.entity_fusion_activation == "sigmoid":
            # https://arxiv.org/abs/2409.04431
            entity_fusion_sigmoid_bias = key_padding_mask.eq(0).sum(dim=-1, keepdim=True)[:, :, None, None]
            entity_fusion_sigmoid_bias = entity_fusion_sigmoid_bias.to(dtype)
            entity_fusion_sigmoid_bias = -torch.log(entity_fusion_sigmoid_bias)

            attention_scores = attention_scores + entity_fusion_sigmoid_bias
            normalized_attention_scores = torch.sigmoid(attention_scores)
        else:
            normalized_attention_scores = nn.functional.softmax(attention_scores, dim=-1)

        context_layer = torch.matmul(normalized_attention_scores, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.config.hidden_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        return (context_layer, orig_attention_scores)


class EntityFusionLayer(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config

        self.entity_embeddings = EntityEmbeddings(config)
        self.entity_fusion_layer = EntityFusionMultiHeadAttention(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.noop_embeddings = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

    def forward(
        self,
        entity_ids: Tensor | None,
        entity_embeds: Tensor | None,
        entity_position_ids: Tensor,
        cls_embeddings: Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        entity_embeddings = self.entity_embeddings(entity_ids, entity_embeds, entity_position_ids)

        batch_size = entity_ids.size(0)
        kv_embeddings = entity_embeddings
        key_padding_mask = entity_ids.eq(0)
        cls_embeddings = cls_embeddings.unsqueeze(1)

        noop_embeddings = self.noop_embeddings.expand(batch_size, 1, -1)

        kv_embeddings = torch.cat([noop_embeddings, kv_embeddings], dim=1)
        noop_padding_mask = torch.zeros(batch_size, 1, device=entity_ids.device, dtype=torch.bool)
        key_padding_mask = torch.cat([noop_padding_mask, key_padding_mask], dim=1)

        entity_embeddings, attention_scores = self.entity_fusion_layer(
            cls_embeddings, kv_embeddings, kv_embeddings, key_padding_mask=key_padding_mask
        )
        entity_embeddings = self.dropout(entity_embeddings)
        output_embeddings = entity_embeddings + cls_embeddings
        output_embeddings = self.LayerNorm(output_embeddings)

        output_embeddings = output_embeddings.squeeze(1)

        return output_embeddings, attention_scores


class KPRMixin:
    def _forward(self, **inputs: dict[str, Tensor]) -> tuple[Tensor] | tuple[Tensor, Tensor] | ModelOutput:
        return_dict = inputs.pop("return_dict", True)

        if self.training:
            query_embeddings = self.encode(**inputs["queries"])
            passage_embeddings = self.encode(**inputs["passages"])

            query_embeddings = self._dist_gather_tensor(query_embeddings)
            passage_embeddings = self._dist_gather_tensor(passage_embeddings)

            scores = self._compute_similarity(query_embeddings, passage_embeddings)
            scores = scores / self.config.similarity_temperature
            scores = scores.view(query_embeddings.size(0), -1)

            ce_target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
            ce_target = ce_target * (passage_embeddings.size(0) // query_embeddings.size(0))
            loss = torch.nn.CrossEntropyLoss(reduction="mean")(scores, ce_target)

            if return_dict:
                return ModelOutput(loss=loss, scores=scores)
            else:
                return (loss, scores)

        else:
            sentence_embeddings = self.encode(**inputs).unsqueeze(1)
            if return_dict:
                return ModelOutput(sentence_embeddings=sentence_embeddings)
            else:
                return (sentence_embeddings,)

    def encode(self, **inputs: dict[str, Tensor]) -> Tensor:
        entity_ids = inputs.pop("entity_ids", None)
        entity_position_ids = inputs.pop("entity_position_ids", None)
        entity_embeds = inputs.pop("entity_embeds", None)

        outputs = getattr(self, self.base_model_prefix)(**inputs)
        output_embeddings = outputs.last_hidden_state[:, 0]

        if self.config.entity_fusion_method != "none":
            output_embeddings, _ = self.entity_fusion_layer(
                entity_ids=entity_ids,
                entity_embeds=entity_embeds,
                entity_position_ids=entity_position_ids,
                cls_embeddings=output_embeddings,
            )
        if self.config.similarity_function == "cosine":
            output_embeddings = F.normalize(output_embeddings, p=2, dim=-1)

        return output_embeddings

    def _dist_gather_tensor(self, t: torch.Tensor) -> torch.Tensor:
        t = t.contiguous()
        tensor_list = [torch.empty_like(t) for _ in range(dist.get_world_size())]
        dist.all_gather(tensor_list, t)

        tensor_list[dist.get_rank()] = t
        gathered_tensor = torch.cat(tensor_list, dim=0)

        return gathered_tensor

    def _compute_similarity(self, query_embeddings: Tensor, passage_embeddings: Tensor) -> Tensor:
        return torch.matmul(query_embeddings, passage_embeddings.transpose(-2, -1))


class KPRModelForBert(BertPreTrainedModel, KPRMixin):
    config_class = KPRConfigForBert

    def __init__(self, config: KPRConfigForBert):
        BertPreTrainedModel.__init__(self, config)

        self.bert = BertModel(config)
        if self.config.entity_fusion_method != "none":
            self.entity_fusion_layer = EntityFusionLayer(config)

        self.post_init()

    def forward(self, *args, **kwargs):
        return self._forward(*args, **kwargs)


class KPRModelForXLMRoberta(XLMRobertaPreTrainedModel, KPRMixin):
    config_class = KPRConfigForXLMRoberta

    def __init__(self, config: KPRConfigForXLMRoberta):
        XLMRobertaPreTrainedModel.__init__(self, config)

        self.roberta = XLMRobertaModel(config)
        if self.config.entity_fusion_method != "none":
            self.entity_fusion_layer = EntityFusionLayer(config)

        self.post_init()

    def forward(self, *args, **kwargs):
        return self._forward(*args, **kwargs)
