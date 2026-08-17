"""Market agent — market intelligence & competitive analysis (spec §5.6).

Reuses the proven research pipeline (Bing search → fetch → synthesize)
with a market-specific synthesis prompt and report templates: SWOT,
trends, competitors.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from ultra.config import Config
from ultra.display import info, ok, step, warn
from ultra.llm import OllamaClient
from ultra.persona import system_prompt
from ultra.tools.researcher import Researcher, Source

MARKET_MODES = {
    "swot": "SWOT analysis: strengths, weaknesses, opportunities, threats for each major player",
    "trends": "market trends: growth drivers, emerging patterns, tech shifts, and adoption signals",
    "competitors": "competitive landscape: key players, positioning, pricing, differentiation",
    "overview": "market overview: size, segments, players, and strategic recommendations",
}

SYSTEM = system_prompt("market analyst") + """

You synthesize market intelligence from provided sources. Rules:
- Cite sources inline as [1], [2] ... matching the source list order.
- NEVER invent a source, citation number, or market figure not in the sources.
- Distinguish verified facts from estimates; label estimates as such.
- End with ONE clear strategic recommendation.
- Markdown headings. Concise but complete."""

SYSTEM_NO_SOURCES = system_prompt("market analyst") + """

No external sources are available. Rules:
- Base the analysis on your own knowledge only.
- DO NOT use [1], [2] or any citation markers — there is nothing to cite.
- Start with the line: "No external sources found — this is based on model knowledge, not web research."
- Label every figure as an estimate.
- End with ONE clear strategic recommendation."""

CITATION_MARKER = re.compile(r"\[\d+(?:-\d+)?\]")
CITATION_LINE = re.compile(r"^\s*(?:Source|Sources?):?\s*\[\d[^\n]*$", re.MULTILINE)


@dataclass
class MarketReport:
    topic: str
    mode: str
    markdown: str
    sources: list[Source] = field(default_factory=list)
    saved_to: str = ""

    @property
    def citation_lines(self) -> str:
        return "\n".join(
            f"{i}. {s.title} — {s.url}" for i, s in enumerate(self.sources, 1)
        )


class MarketAgent:
    def __init__(self, client: OllamaClient, config: Config):
        self.client = client
        self.config = config
        self.researcher = Researcher(config)

    def run(self, topic: str, mode: str = "overview") -> MarketReport:
        mode = mode if mode in MARKET_MODES else "overview"
        step(1, f"Searching market intelligence ({mode})")
        sources = self.researcher.deep_search(topic)
        if not sources:
            warn("no web results — building analysis from model knowledge only")
        else:
            ok(f"{len(sources)} sources found")

        step(2, "Fetching sources")
        enriched = []
        for src in sources[:4]:
            content = self.researcher.fetch(src.url)
            if content:
                src.content = content
            enriched.append(src)
        info(f"{len([s for s in enriched if s.content])} pages extracted")

        step(3, "Synthesizing")
        if enriched:
            source_block = "\n\n".join(
                f"[{i}] {s.title}\n{s.text(2500)}"
                for i, s in enumerate(enriched, 1)
            )
            prompt = f"""Market topic: {topic}
Mode: {MARKET_MODES[mode]}

SOURCES:
{source_block}

Write the market analysis in markdown with inline citations. Cite ONLY the sources above."""
            markdown = self.client.generate(prompt, system=SYSTEM,
                                            max_tokens=4096, temperature=0.4,
                                            task_type="research")
        else:
            prompt = f"""Market topic: {topic}
Mode: {MARKET_MODES[mode]}

No external sources available. Write the analysis from your own knowledge."""
            markdown = self.client.generate(prompt, system=SYSTEM_NO_SOURCES,
                                            max_tokens=4096, temperature=0.4,
                                            task_type="research")
        # citation honesty — same safety net as research
        if not enriched:
            markdown = CITATION_LINE.sub("", markdown)
            markdown = CITATION_MARKER.sub("", markdown)
        return MarketReport(topic=topic, mode=mode, markdown=markdown,
                            sources=enriched)

    def save(self, report: MarketReport, out_dir: Path | None = None) -> str:
        out_dir = out_dir or (self.config.projects_dir / "market")
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        slug = "".join(c if c.isalnum() else "-" for c in report.topic.lower())[:40]
        path = out_dir / f"{slug}-{stamp}.md"
        body = f"# Market: {report.topic}\n\n*Mode: {report.mode}*\n\n{report.markdown}\n"
        if report.sources:
            body += f"\n---\n\n## Sources\n{report.citation_lines}\n"
        else:
            body += "\n---\n\n*No external sources — analysis is based on model knowledge.*\n"
        path.write_text(body, encoding="utf-8")
        report.saved_to = str(path)
        return str(path)
