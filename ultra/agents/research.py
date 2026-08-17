"""R&D research agent.

Multi-angle search → fetch top sources → synthesize a structured report
with inline citations. All LLM synthesis happens locally via Ollama; only
web fetching uses the network (no API keys).

Citation honesty: when no real sources are found, the report must say so
instead of inventing [1] markers. Fabricated citations are stripped
post-generation as a safety net.
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

RESEARCH_MODES = {
    "deep": "deep research report with multiple angles",
    "compare": "side-by-side comparison with a feature matrix",
    "feasibility": "feasibility analysis with risk assessment and timeline",
    "competitive": "SWOT-style competitive analysis",
}

SYSTEM = system_prompt("research analyst") + """

You synthesize research from provided sources. Rules:
- Cite sources inline as [1], [2] ... matching the source list order.
- NEVER invent a source or citation number that is not in the source list.
- Be specific and factual — never invent facts not in the sources.
- Give ONE clear recommendation at the end.
- Markdown headings. Concise but complete."""

SYSTEM_NO_SOURCES = system_prompt("research analyst") + """

No external sources are available for this topic. Rules:
- Base the report on your own knowledge only.
- DO NOT use [1], [2] or any citation markers — there is nothing to cite.
- Start the report with the line: "No external sources found — this is based on model knowledge, not web research."
- Be honest about uncertainty.
- Give ONE clear recommendation at the end.
- Markdown headings. Concise but complete."""

CITATION_MARKER = re.compile(r"\[\d+(?:-\d+)?\]")
CITATION_LINE = re.compile(r"^\s*(?:Source|Sources?):?\s*\[\d[^\n]*$", re.MULTILINE)


def sanitize_citations(markdown: str, n_sources: int) -> str:
    """Strip fabricated citation markers.

    With no real sources, every [N] marker and "Source: [N]" line is
    removed. With real sources, markers pointing past the source count are
    removed too (those are invented).
    """
    if n_sources == 0:
        # drop citation LINES first (they still contain the [N] marker),
        # then strip any remaining inline markers
        text = CITATION_LINE.sub("", markdown)
        text = CITATION_MARKER.sub("", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
    # with sources: drop only out-of-range markers
    def _drop(match: re.Match) -> str:
        nums = [int(x) for x in re.findall(r"\d+", match.group(0))]
        return "" if any(n > n_sources for n in nums) else match.group(0)
    return CITATION_MARKER.sub(_drop, markdown)


@dataclass
class ResearchReport:
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


class ResearchAgent:
    def __init__(self, client: OllamaClient, config: Config):
        self.client = client
        self.config = config
        self.researcher = Researcher(config)

    def run(self, topic: str, mode: str = "deep") -> ResearchReport:
        mode = mode if mode in RESEARCH_MODES else "deep"
        step(1, f"Searching ({mode})")
        sources = self.researcher.deep_search(topic)
        if not sources:
            warn("no web results — building report from model knowledge only")
        else:
            ok(f"{len(sources)} sources found")

        step(2, "Fetching sources")
        enriched = []
        for src in sources[:4]:  # fetch top 4 for content
            content = self.researcher.fetch(src.url)
            if content:
                src.content = content
            enriched.append(src)
        info(f"{len([s for s in enriched if s.content])} pages extracted")

        step(3, "Synthesizing")
        mode_instruction = RESEARCH_MODES[mode]
        has_sources = bool(enriched)
        if has_sources:
            source_block = "\n\n".join(
                f"[{i}] {s.title}\n{s.text(2500)}"
                for i, s in enumerate(enriched, 1)
            )
            prompt = f"""Topic: {topic}
Mode: {mode_instruction}

SOURCES:
{source_block}

Write the report in markdown with inline citations. Cite ONLY the sources above."""
            markdown = self.client.generate(prompt, system=SYSTEM,
                                            max_tokens=4096, temperature=0.4)
        else:
            prompt = f"""Topic: {topic}
Mode: {mode_instruction}

No external sources are available. Write the report from your own knowledge."""
            markdown = self.client.generate(prompt, system=SYSTEM_NO_SOURCES,
                                            max_tokens=4096, temperature=0.4)
        markdown = sanitize_citations(markdown, len(enriched))
        return ResearchReport(topic=topic, mode=mode, markdown=markdown,
                              sources=enriched)

    def save(self, report: ResearchReport, out_dir: Path | None = None) -> str:
        out_dir = out_dir or (self.config.projects_dir / "research")
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        slug = "".join(c if c.isalnum() else "-" for c in report.topic.lower())[:40]
        path = out_dir / f"{slug}-{stamp}.md"
        body = f"# {report.topic}\n\n*Mode: {report.mode}*\n\n{report.markdown}\n"
        if report.sources:
            body += f"\n---\n\n## Sources\n{report.citation_lines}\n"
        else:
            body += "\n---\n\n*No external sources — report is based on model knowledge.*\n"
        path.write_text(body, encoding="utf-8")
        report.saved_to = str(path)
        return str(path)
