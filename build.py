#!/usr/bin/env python3
"""
build.py — The Tape

Reads feeds.txt, fetches each feed, tags stories with classify.py, renders
template.html, writes index.html, bumps the issue number in state.json.

Design notes:
  * Every network call is wrapped. A dead feed, a rate-limited price API, or a
    missing Gemini key degrades the page gracefully rather than failing the run.
    A news site that doesn't build is worse than one missing a section.
  * No story is invented. If a section has nothing in it, it is omitted from
    the page entirely rather than padded.
"""

from __future__ import annotations

import html
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import feedparser
import requests

from classify import classify_story

ROOT = Path(__file__).parent
FEEDS_FILE = ROOT / "feeds.txt"
TEMPLATE = ROOT / "template.html"
STATE_FILE = ROOT / "state.json"
OUTPUT = ROOT / "index.html"

MAX_PER_SOURCE = 2          # stops one prolific feed owning the page
MAX_HEADLINES = 10
MAX_PROJECTS = 5
MAX_DEGEN = 4
MAX_QUICK_HITS = 8
FEED_TIMEOUT = 20

HEADLINE_TAGS = {"MARKET", "PROTOCOL", "REGULATION", "SECURITY", "GENERAL"}

# Base rate shown permanently in the degen section header. Source:
# arXiv 2607.02823 (preprint, not peer-reviewed) reports a 0.198% pump.fun
# graduation rate; DEXTools reported ~0.26% for mid-June 2026. If you change
# this string, keep a real number with a real source behind it.
DEGEN_NOTE = (
    "Base rate: roughly 0.2&ndash;0.6% of pump.fun launches ever graduate. "
    "Listed here because they moved, not because they are worth buying."
)


# --------------------------------------------------------------------------
# Feeds
# --------------------------------------------------------------------------

def load_feeds() -> list[dict]:
    feeds = []
    if not FEEDS_FILE.exists():
        sys.exit("feeds.txt not found")

    for line_no, raw in enumerate(FEEDS_FILE.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 4:
            print(f"  ! feeds.txt line {line_no}: expected 4 fields, skipping")
            continue
        url, name, trust, section = parts[:4]
        try:
            trust_val = int(trust)
        except ValueError:
            print(f"  ! feeds.txt line {line_no}: bad trust value '{trust}', defaulting to 5")
            trust_val = 5
        feeds.append({"url": url, "name": name, "trust": trust_val, "section": section})
    return feeds


def fetch(feed: dict) -> list[dict]:
    """Fetch one feed. Never raises — a dead feed just contributes nothing."""
    try:
        resp = requests.get(
            feed["url"],
            timeout=FEED_TIMEOUT,
            headers={"User-Agent": "TheTape/1.0 (+https://mod108.github.io/The-Tape)"},
        )
        resp.raise_for_status()
        parsed = feedparser.parse(resp.content)
    except Exception as exc:
        print(f"  ! {feed['name']}: {type(exc).__name__} — skipped")
        return []

    if not parsed.entries:
        print(f"  ! {feed['name']}: parsed but returned 0 entries")
        return []

    stories = []
    for entry in parsed.entries[:15]:
        title = clean(getattr(entry, "title", ""))
        if not title:
            continue
        stories.append({
            "title": title,
            "url": getattr(entry, "link", ""),
            "summary": clean(getattr(entry, "summary", ""))[:400],
            "published": to_dt(entry),
            "source": feed["name"],
            "trust": feed["trust"],
        })
    print(f"  + {feed['name']}: {len(stories)}")
    return stories


def clean(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text or "")
    return html.unescape(text).strip()


def to_dt(entry) -> datetime:
    for field in ("published_parsed", "updated_parsed"):
        value = getattr(entry, field, None)
        if value:
            try:
                return datetime(*value[:6], tzinfo=timezone.utc)
            except (TypeError, ValueError):
                pass
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------
# Dedup — same story across outlets
# --------------------------------------------------------------------------

def signature(title: str) -> frozenset[str]:
    words = re.findall(r"[a-z0-9]+", title.lower())
    stop = {"the", "a", "an", "to", "of", "in", "on", "for", "and", "as", "is", "at", "by"}
    return frozenset(w for w in words if w not in stop and len(w) > 2)


def dedupe(stories: list[dict]) -> list[dict]:
    """Keep the highest-trust version of each story."""
    stories = sorted(stories, key=lambda s: (-s["trust"], -s["published"].timestamp()))
    kept: list[dict] = []
    for story in stories:
        sig = signature(story["title"])
        if not sig:
            continue
        duplicate = False
        for existing in kept:
            overlap = sig & existing["_sig"]
            if len(overlap) / max(len(sig), 1) > 0.6:
                duplicate = True
                break
        if not duplicate:
            story["_sig"] = sig
            kept.append(story)
    return kept


def cap_per_source(stories: list[dict], limit: int = MAX_PER_SOURCE) -> list[dict]:
    counts: dict[str, int] = {}
    out = []
    for story in stories:
        n = counts.get(story["source"], 0)
        if n >= limit:
            continue
        counts[story["source"]] = n + 1
        out.append(story)
    return out


# --------------------------------------------------------------------------
# Optional Gemini summaries — absent key is a normal, silent fallback
# --------------------------------------------------------------------------

def summarise(story: dict) -> str:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    fallback = story["summary"][:220]
    if not key:
        return fallback
    try:
        resp = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/models/"
            "gemini-2.0-flash:generateContent",
            params={"key": key},
            json={"contents": [{"parts": [{
                "text": (
                    "Summarise this crypto news story in one flat, factual sentence. "
                    "No hype, no adjectives, no price speculation. If the story is "
                    "promotional rather than news, say so.\n\n"
                    f"Headline: {story['title']}\n\n{story['summary']}"
                )
            }]}]},
            timeout=25,
        )
        resp.raise_for_status()
        data = resp.json()
        return clean(data["candidates"][0]["content"]["parts"][0]["text"]) or fallback
    except Exception as exc:
        print(f"  ! Gemini failed on '{story['title'][:40]}': {type(exc).__name__}")
        return fallback


# --------------------------------------------------------------------------
# Render
# --------------------------------------------------------------------------

def story_html(story: dict) -> str:
    promo = " story--promo" if story.get("is_promo") else ""
    tag = story["tag"]
    stamp = story["published"].strftime("%d %b %Y &middot; %H:%M UTC").upper()
    return f"""
      <article class="story{promo}">
        <p class="meta">
          <span class="tag tag--{tag.lower()}">{tag}</span>
          <span class="timestamp">{stamp}</span>
          <span class="source">{html.escape(story['source'])}</span>
        </p>
        <h3><a href="{html.escape(story['url'])}">{html.escape(story['title'])}</a></h3>
        <p class="standfirst">{html.escape(story['blurb'])}</p>
      </article>"""


def section_html(title: str, stories: list[dict], note: str = "") -> str:
    if not stories:
        return ""
    note_html = f'<p class="degen-note">{note}</p>' if note else ""
    body = "\n".join(story_html(s) for s in stories)
    return f"""
    <section class="section">
      <h2 class="section-head">{title}</h2>
      {note_html}
      {body}
    </section>"""


def quick_hits_html(stories: list[dict]) -> str:
    if not stories:
        return ""
    items = "\n".join(
        f'        <li><span class="timestamp">{s["published"].strftime("%d %b").upper()}</span>'
        f'<a href="{html.escape(s["url"])}">{html.escape(s["title"])}</a></li>'
        for s in stories
    )
    return f"""
    <section class="section">
      <h2 class="section-head">Quick hits</h2>
      <ul class="quick-hits">
{items}
      </ul>
    </section>"""


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except json.JSONDecodeError:
            print("  ! state.json unreadable, restarting issue count")
    return {"issue": 0}


def main() -> None:
    print("The Tape — build starting")

    feeds = load_feeds()
    print(f"{len(feeds)} feeds configured")

    raw: list[dict] = []
    for feed in feeds:
        raw.extend(fetch(feed))

    if not raw:
        sys.exit("No stories fetched from any feed — refusing to publish an empty issue")

    print(f"{len(raw)} stories fetched")

    for story in raw:
        verdict = classify_story(story["title"], story["summary"], story["trust"])
        story["tag"] = verdict.tag
        story["is_promo"] = verdict.is_promo

    stories = dedupe(raw)
    print(f"{len(stories)} after dedup")

    # Promo-flagged low-trust stories drop out entirely; flagged high-trust
    # ones stay but render demoted (see .story--promo in theme.css).
    stories = [s for s in stories if not (s["is_promo"] and s["trust"] <= 6)]

    stories.sort(key=lambda s: -s["published"].timestamp())

    headlines = cap_per_source([s for s in stories if s["tag"] in HEADLINE_TAGS])[:MAX_HEADLINES]
    projects = [s for s in stories if s["tag"] == "PROJECT"][:MAX_PROJECTS]
    degen = [s for s in stories if s["tag"] == "DEGEN"][:MAX_DEGEN]

    featured = headlines + projects + degen
    used = {id(s) for s in featured}
    quick = [s for s in stories if id(s) not in used][:MAX_QUICK_HITS]

    for story in featured:
        story["blurb"] = summarise(story)

    state = load_state()
    state["issue"] = state.get("issue", 0) + 1
    now = datetime.now(timezone.utc)

    ticker = " &nbsp;///&nbsp; ".join(
        html.escape(s["title"]) for s in headlines[:5]
    ) or "The Tape"

    page = TEMPLATE.read_text(encoding="utf-8")
    page = (page
            .replace("{{ISSUE}}", f"{state['issue']:03d}")
            .replace("{{DATE}}", now.strftime("%A %d %B %Y"))
            .replace("{{BUILT}}", now.strftime("%d %b %Y &middot; %H:%M UTC"))
            .replace("{{TICKER}}", ticker)
            .replace("{{HEADLINES}}", section_html("Headlines", headlines))
            .replace("{{PROJECTS}}", section_html("New projects", projects))
            .replace("{{DEGEN}}", section_html("Degen corner", degen, DEGEN_NOTE))
            .replace("{{QUICK_HITS}}", quick_hits_html(quick)))

    OUTPUT.write_text(page, encoding="utf-8")
    STATE_FILE.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

    print(f"Issue {state['issue']:03d}: {len(headlines)} headlines, "
          f"{len(projects)} projects, {len(degen)} degen, {len(quick)} quick hits")


if __name__ == "__main__":
    main()
