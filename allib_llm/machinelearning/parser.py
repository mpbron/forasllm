import re
from typing import Optional, Sequence, Tuple

from parsec import ParseError, generate, regex, separated, sepBy, string


def parse_llm_output(text: str) -> Optional[Tuple[str, str]]:
    # Define regular expressions to match REASONING and ANSWER
    reasoning_pattern = r"REASONING:\s*(.*?)\s*ANSWER:\s*(.*?)\s*"

    # Search for the pattern in the input text
    match = re.search(reasoning_pattern, text, re.DOTALL)

    # If a match is found, extract the data and store it in the dictionary
    if match:
        reasoning = match.group(1).strip()
        answer = text[match.regs[2][1] :].strip()
        return reasoning, answer
    return None


def parse_llm_qa_output(raw_str: str) -> Tuple[str, str, str]:
    # define a pattern for each part
    evidence_pattern = r"EVIDENCE:\s*(.*?)\s*ANSWER:"
    reasoning_pattern = r"REASONING:\s*(.*?)\s*EVIDENCE:"
    answer_pattern = r"ANSWER:\s*(.*)"

    # use regex to find the matches
    evidence_match = re.search(evidence_pattern, raw_str, re.DOTALL)
    reasoning_match = re.search(reasoning_pattern, raw_str, re.DOTALL)
    answer_match = re.search(answer_pattern, raw_str, re.DOTALL)

    # if a match is found, extract it, otherwise set as None
    evidence = evidence_match.group(1).strip() if evidence_match else ""
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    answer = answer_match.group(1).strip() if answer_match else ""

    return evidence, reasoning, answer


spaces = regex(r"\s*", re.MULTILINE)
name = regex(r"[_a-zA-Z][_a-zA-Z0-9]*")
string_chars = regex(r"[^\"\n]*")


def pQuotes(pFunc):
    quotes = string('"') >> pFunc << string('"')
    db_quotes = string("'") >> pFunc << string("'")
    return quotes ^ db_quotes


def pListBracketes(pFunc):
    return string("[") >> pFunc << string("]")


def pBrackets(pFunc):
    return string("(") >> pFunc << string(")")


def pCommaSpace():
    return spaces >> string(",") << spaces


def pBullit():
    return spaces >> string("-") << spaces


@generate
def pStrListParser():
    elems = yield pListBracketes(sepBy(pQuotes(string_chars), pCommaSpace()))
    return list(elems)


@generate
def pBullitListParse():
    yield pBullit()
    elems = yield sepBy(pQuotes(string_chars), pBullit())
    return list(elems)


@generate
def pNormalListParser():
    elems = yield sepBy(pQuotes(string_chars), pCommaSpace())
    return list(elems)

@generate
def pEvidenceList():
    listparser = pStrListParser ^ pBullitListParse ^ pNormalListParser
    res = yield listparser
    return res

def tryparselist(text: str) -> Optional[Sequence[str]]:
    try:
        parsed = pEvidenceList.parse(text)
    except ParseError:
        return None
    return parsed