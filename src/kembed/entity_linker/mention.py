from dataclasses import dataclass


@dataclass
class BaseMention:
    kb_id: str | None
    text: str
    start: int
    end: int

    @property
    def span(self) -> tuple[int, int]:
        return self.start, self.end

    def __repr__(self):
        return f"<Mention {self.text} -> {self.kb_id}>"
