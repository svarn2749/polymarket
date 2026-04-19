"""Derive a coarse topic label from a Polymarket slug.

No API roundtrip — just regex rules tuned on observed slug patterns. Meant as
a research aid, not ground truth. Markets that don't match any rule return
"other".
"""
from __future__ import annotations

import re

# Order matters: earlier rules win. Keep more-specific before more-generic.
_RULES: list[tuple[str, re.Pattern[str]]] = [
    (
        "crypto",
        re.compile(
            r"\b(bitcoin|btc|ethereum|eth|solana|\bsol\b|xrp|dogecoin|doge|cardano|"
            r"crypto|stablecoin|tether|usdc|usdt|memecoin|altcoin|nft)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "sports",
        re.compile(
            r"\b(nba|nfl|mlb|nhl|mls|premier-?league|wimbledon|us-?open|french-?open|"
            r"australian-?open|grand-?slam|world-?cup|super-?bowl|world-?series|stanley-?cup|"
            r"champion|playoff|ufc|mma|boxing|dota|league-of-legends|lol-|csgo|valorant|"
            r"soccer|tennis|basketball|football|hockey|baseball|cricket|golf|pga|masters-tournament|"
            r"olympic|arsenal|liverpool|manchester|chelsea|madrid|barcelona|lakers|celtics|"
            r"warriors|cowboys|patriots|yankees|dodgers|bucks|heat|nuggets|thunder|mavericks|"
            r"grizzlies|76ers|knicks|clippers|rockets|sixers|kings|cavaliers|bulls|raptors|"
            r"mammoth|hurricanes|blackhawks|bruins|canucks|western-conference|eastern-conference)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "politics",
        re.compile(
            r"\b(trump|biden|harris|obama|clinton|rubio|desantis|vance|musk|kennedy|"
            r"president|presidential|election|congress|senate|governor|prime-?minister|\bpm\b|"
            r"putin|xi-?jinping|netanyahu|zelensky|erdogan|macron|starmer|"
            r"democratic|republican|primary|caucus|electoral|parliament|"
            r"iran|israel|ukraine|russia|china|north-?korea|nato|\beu\b|brexit|"
            r"supreme-?court|scotus|impeach|indict|investigation|fbi|cia|"
            r"ceasefire|war|invasion|sanctions|treaty|peace-?deal)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "econ",
        re.compile(
            r"\b(recession|cpi|ppi|inflation|deflation|gdp|fed-?rate|interest-?rate|"
            r"unemployment|jobs-?report|payroll|s-?and-?p-?500|sp500|dow|nasdaq|"
            r"stock-?market|gold-?price|silver-?price|oil-?price|wti|brent|copper)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "entertainment",
        re.compile(
            r"\b(oscar|emmy|grammy|tony-?award|box-?office|movie|film|album|song|"
            r"bachelor|bachelorette|love-?island|survivor|reality-?tv|award|drake|taylor-?swift|"
            r"beyonce|kanye|rihanna|kardashian|jenner|streaming|netflix|disney|marvel|dc)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "weather",
        re.compile(
            r"\b(hurricane|tornado|storm|rainfall|snowfall|temperature|weather|climate|"
            r"earthquake|wildfire|heat-?wave)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "tech",
        re.compile(
            r"\b(openai|anthropic|claude|chatgpt|gpt-|llama|gemini|apple|google|microsoft|"
            r"nvidia|tesla|spacex|meta|facebook|twitter|\bx\b-com|amazon|\baws\b|ipo|stock-split)\b",
            re.IGNORECASE,
        ),
    ),
]


def derive_topic(slug: str | None, question: str | None = None) -> str:
    """Return a best-guess topic bucket for a market.

    Checks the slug first, then falls back to question text. Returns "other"
    when nothing matches.
    """
    blob = " ".join(x for x in (slug, question) if x)
    if not blob:
        return "other"
    for label, pattern in _RULES:
        if pattern.search(blob):
            return label
    return "other"
