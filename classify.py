"""
classify.py — story tagging and promo screening for The Tape.

WHY THIS EXISTS
---------------
On The Wire, every story on the front page came back tagged LAUNCH, including
an Ars Technica piece on radiologists and a Google home-decor post. That is the
signature of a keyword tagger with no confidence floor: it scores every category,
finds no strong match, and falls through to whichever bucket is first.

Two fixes here:

  1. A confidence FLOOR. If nothing scores above the threshold, the story is
     tagged GENERAL rather than being forced into a category. Honest "unknown"
     beats a confident wrong label.
  2. Ordering by score, not by dict order, with an explicit tie-break.

NOTE: I have not seen The Wire's current build.py, so this is written fresh as a
drop-in module rather than as a patch to your existing function. Wire it in by
replacing your tagging call with:

    from classify import classify_story
    tag, confidence, is_promo = classify_story(title, summary, source_trust)

If your existing classifier already does something smarter than this, keep yours
and just add the floor.
"""

from __future__ import annotations

import re
from typing import NamedTuple

# Score a story must reach before we accept a category label at all.
# Tune by running against a day of real headlines and eyeballing the output.
CONFIDENCE_FLOOR = 2.0

# Sections. GENERAL is the honest fallback, not a category.
SECTIONS = ("MARKET", "PROTOCOL", "REGULATION", "SECURITY", "PROJECT", "DEGEN", "GENERAL")

# Weighted signals. Weight reflects how strongly the term implies the category —
# a term that appears across all crypto writing scores low even if it is common.
SIGNALS: dict[str, list[tuple[str, float]]] = {
    "MARKET": [
        (r"\b(rally|selloff|sell-off|plunge|surge|all-time high|ATH)\b", 2.0),
        (r"\b(ETF|inflows?|outflows?|open interest|liquidations?)\b", 2.5),
        (r"\b(bitcoin|ether|BTC|ETH|SOL)\s+(price|falls?|rises?|jumps?|drops?)", 2.5),
        (r"\b(market cap|trading volume|funding rate)\b", 1.5),
    ],
    "PROTOCOL": [
        (r"\b(mainnet|testnet|hard fork)\b", 3.0),
        # "upgrade" alone is a generic English word — it tagged a Google
        # home-decor headline as PROTOCOL in testing. Only score it high
        # when it sits next to a network noun.
        (r"\b(network|protocol|chain|client|node)\s+upgrade\b", 3.0),
        (r"\bupgrade\b", 1.0),
        (r"\b(layer[- ]?2|L2|rollup|zk|validator|staking|consensus)\b", 2.0),
        (r"\b(EIP|SIP|BIP)[- ]?\d+", 3.5),
        (r"\b(bridge|interoperability|throughput|finality)\b", 1.5),
    ],
    "REGULATION": [
        (r"\b(SEC|CFTC|FCA|MiCA|Treasury|OFAC|IRS|HMRC)\b", 3.0),
        (r"\b(lawsuit|sues?|settle(?:ment|s)?|subpoena|indict(?:ed|ment))\b", 2.5),
        (r"\b(regulat\w+|complian\w+|licen[cs]\w+|sanctions?)\b", 2.0),
        (r"\b(bill|legislation|ruling|court|judge)\b", 1.5),
    ],
    "SECURITY": [
        (r"\b(hack(?:ed|er)?|exploit(?:ed)?|breach|drained?)\b", 3.5),
        (r"\b(vulnerabilit\w+|attack vector|reentrancy|oracle manipulation)\b", 3.0),
        (r"\$\d[\d.,]*\s*(million|billion|m|bn)\s+(stolen|lost|drained)", 4.0),
        (r"\b(rug ?pull|honeypot|scam|phishing)\b", 2.5),
    ],
    "PROJECT": [
        (r"\b(raises?|raised|funding round|Series [A-D]|seed round)\b", 3.5),
        (r"\b(launch(?:es|ed)?|unveils?|debuts?|goes live)\b", 2.0),
        (r"\b(token generation event|TGE|airdrop|IDO|ICO)\b", 3.0),
        (r"\b(led by|backed by|investors? include)\b", 2.0),
    ],
    "DEGEN": [
        (r"\b(meme ?coin|memecoin|pump\.fun|bonding curve)\b", 4.0),
        (r"\b(DOGE|SHIB|PEPE|BONK|WIF)\b", 2.5),
        (r"\b(degen|ape[ds]?\s+in|moon(?:ing|ed)?)\b", 2.0),
    ],
}

# Sponsored / promotional tells. Crypto feeds carry far more of these than the
# AI feeds The Wire pulls, so this screen matters more here.
PROMO_PATTERNS = [
    r"\b(sponsored|press release|partner content|paid post)\b",
    r"\b(don'?t miss|last chance|presale|pre-sale)\b",
    r"\b(best|top)\s+\d+\s+(coins?|tokens?|altcoins?)\s+(to buy|for)\b",
    r"\b(could|will|set to)\s+(explode|100x|10x|skyrocket)\b",
    r"\b(next|new)\s+(bitcoin|ethereum|solana)\b",
    r"\bprice prediction\b",
]

# Trust at or below this gets the promo screen applied strictly.
LOW_TRUST_CEILING = 6


class Verdict(NamedTuple):
    tag: str
    confidence: float
    is_promo: bool


def _score(text: str, patterns: list[tuple[str, float]]) -> float:
    total = 0.0
    for pattern, weight in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            total += weight
    return total


def looks_promotional(text: str, source_trust: int) -> bool:
    """True if the story reads as paid placement or shill content.

    Applied to every source. On low-trust sources a single hit is enough;
    on high-trust sources we require two, because reputable outlets do
    legitimately write about presales and price moves.
    """
    hits = sum(1 for p in PROMO_PATTERNS if re.search(p, text, flags=re.IGNORECASE))
    if source_trust <= LOW_TRUST_CEILING:
        return hits >= 1
    return hits >= 2


def classify_story(title: str, summary: str = "", source_trust: int = 5) -> Verdict:
    """Tag a story, or return GENERAL if nothing clears the confidence floor.

    The title is weighted double: headlines are written to signal the story
    type, summaries wander.
    """
    text = f"{title} {title} {summary}".strip()

    scores = {section: _score(text, patterns) for section, patterns in SIGNALS.items()}
    best_section, best_score = max(scores.items(), key=lambda kv: (kv[1], kv[0]))

    runner_up = sorted(scores.values(), reverse=True)[1] if len(scores) > 1 else 0.0

    # THE FLOOR — this is the fix for the everything-is-LAUNCH bug.
    if best_score < CONFIDENCE_FLOOR:
        return Verdict("GENERAL", best_score, looks_promotional(text, source_trust))

    # Ambiguity guard: if two categories score nearly the same, we are guessing.
    if runner_up > 0 and (best_score - runner_up) < 1.0:
        return Verdict("GENERAL", best_score - runner_up, looks_promotional(text, source_trust))

    return Verdict(best_section, best_score, looks_promotional(text, source_trust))


if __name__ == "__main__":
    # Sanity cases, including the two that The Wire got wrong.
    samples = [
        ("AI won't replace radiologists, but it will change their jobs", "", 9),
        ("5 ways to upgrade your home decor with Google Search", "", 7),
        ("Curve Finance exploited for $8 million in reentrancy attack", "", 9),
        ("SEC drops case against DeFi protocol after two-year fight", "", 8),
        ("Ethereum's Fusaka upgrade goes live on mainnet", "", 9),
        ("Top 5 altcoins to buy before they explode this month", "", 4),
        ("Solana DePIN startup raises $40M Series B led by Paradigm", "", 9),
    ]
    for title, summary, trust in samples:
        v = classify_story(title, summary, trust)
        flag = "  [PROMO]" if v.is_promo else ""
        print(f"{v.tag:<11} {v.confidence:>5.1f}  {title[:58]}{flag}")
