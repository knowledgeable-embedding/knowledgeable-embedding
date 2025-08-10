from transformers import AutoConfig, AutoModel

from .configuration import KPRConfigForBert, KPRConfigForXLMRoberta
from .modeling import KPRModelForBert, KPRModelForXLMRoberta

AutoConfig.register(KPRConfigForBert.model_type, KPRConfigForBert)
AutoConfig.register(KPRConfigForXLMRoberta.model_type, KPRConfigForXLMRoberta)

AutoModel.register(KPRConfigForBert, KPRModelForBert)
AutoModel.register(KPRConfigForXLMRoberta, KPRModelForXLMRoberta)
