from dataclasses import dataclass

from refined.inference.processor import Refined

from .mention import BaseMention


@dataclass
class RefinedELMention(BaseMention):
    score: float


class RefinedEntityLinker:
    def __init__(self, refined: Refined):
        self._refined = refined

    @staticmethod
    def load(
        model_name: str = "wikipedia_model",
        entity_set: str = "wikipedia",
        device: str | None = None,
        use_precomputed_descriptions: bool = True,
    ) -> "RefinedEntityLinker":
        refined = Refined.from_pretrained(
            model_name=model_name,
            entity_set=entity_set,
            device=device,
            use_precomputed_descriptions=use_precomputed_descriptions,
        )
        return RefinedEntityLinker(refined)

    def detect_mentions(self, text: str) -> list[RefinedELMention]:
        spans = self._refined.process_text(text)
        ret = [
            RefinedELMention(
                kb_id=span.predicted_entity.wikidata_entity_id,
                text=span.text,
                start=span.start,
                end=span.start + span.ln,
                score=span.entity_linking_model_confidence_score,
            )
            for span in spans
        ]
        return ret

    def detect_mentions_batch(self, texts: list[str]) -> list[list[RefinedELMention]]:
        docs = self._refined.process_text_batch(texts)
        ret = []
        for doc in docs:
            mentions = [
                RefinedELMention(
                    kb_id=span.predicted_entity.wikidata_entity_id,
                    text=span.text,
                    start=span.start,
                    end=span.start + span.ln,
                    score=span.entity_linking_model_confidence_score,
                )
                for span in doc.spans
            ]
            ret.append(mentions)
        return ret
