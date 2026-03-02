from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Memory:
    short_term: list[str] = field(default_factory=list)
    long_term: list[str] = field(default_factory=list)
    short_term_limit: int = 32
    long_term_limit: int = 64
    write_strategy: str = "append"

    def observe(self, item: str) -> None:
        if (
            self.write_strategy == "forgetful"
            and len(self.short_term) >= self.short_term_limit
        ):
            self.short_term.pop(0)
        self.short_term.append(item)
        if len(self.short_term) > self.short_term_limit:
            overflow = len(self.short_term) - self.short_term_limit
            self.short_term = self.short_term[overflow:]

    def consolidate(self, rng) -> None:
        if not self.short_term:
            return

        if self.write_strategy == "compress":
            take = min(6, len(self.short_term))
            snippet = " | ".join(self.short_term[-take:])
            self.long_term.append(snippet)
            self.short_term = self.short_term[:-take]
        else:
            take = min(3, len(self.short_term))
            for s in self.short_term[-take:]:
                self.long_term.append(s)
            self.short_term = self.short_term[:-take]

        if len(self.long_term) > self.long_term_limit:
            overflow = len(self.long_term) - self.long_term_limit
            self.long_term = self.long_term[overflow:]

    def recall(self, query: str, limit: int = 6) -> list[str]:
        if not query:
            return []
        query_l = query.lower()
        matches: list[str] = []
        for item in reversed(self.long_term + self.short_term):
            if query_l in item.lower():
                matches.append(item)
                if len(matches) >= limit:
                    break
        return matches
