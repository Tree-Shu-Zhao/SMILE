"""VizWiz answer normalization and demonstration labels.

Shared by the scorer (evaluate_results.py) and the dataset loader (load_data.py).
"""

import re
from collections import Counter
from typing import Dict, List, Optional

# Normalization and naming intentionally follow the official VizWiz evaluation API (vqaEval.py):
# https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_data/API.zip

punct = [";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!"]

manualMap = {"none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}

articles = ["a", "an", "the"]

periodStrip = re.compile(r"(?!<=\d)(\.)(?!\d)")
commaStrip = re.compile(r"(\d)(\,)(\d)")

contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't", "couldn'tve": "couldn’t’ve", "couldnt’ve": "couldn’t’ve", "didnt": "didn’t", "doesnt": "doesn’t", "dont": "don’t", "hadnt": "hadn’t", "hadnt’ve": "hadn’t’ve", "hadn'tve": "hadn’t’ve", "hasnt": "hasn’t", "havent": "haven’t", "hed": "he’d", "hed’ve": "he’d’ve", "he’dve": "he’d’ve", "hes": "he’s", "howd": "how’d", "howll": "how’ll", "hows": "how’s", "Id’ve": "I’d’ve", "I’dve": "I’d’ve", "Im": "I’m", "Ive": "I’ve", "isnt": "isn’t", "itd": "it’d", "itd’ve": "it’d’ve", "it’dve": "it’d’ve", "itll": "it’ll", "let’s": "let’s", "maam": "ma’am", "mightnt": "mightn’t", "mightnt’ve": "mightn’t’ve", "mightn’tve": "mightn’t’ve", "mightve": "might’ve", "mustnt": "mustn’t", "mustve": "must’ve", "neednt": "needn’t", "notve": "not’ve", "oclock": "o’clock", "oughtnt": "oughtn’t", "ow’s’at": "’ow’s’at", "’ows’at": "’ow’s’at", "’ow’sat": "’ow’s’at", "shant": "shan’t", "shed’ve": "she’d’ve", "she’dve": "she’d’ve", "she’s": "she’s", "shouldve": "should’ve", "shouldnt": "shouldn’t", "shouldnt’ve": "shouldn’t’ve", "shouldn’tve": "shouldn’t’ve", "somebody’d": "somebodyd", "somebodyd’ve": "somebody’d’ve", "somebody’dve": "somebody’d’ve", "somebodyll": "somebody’ll", "somebodys": "somebody’s", "someoned": "someone’d", "someoned’ve": "someone’d’ve", "someone’dve": "someone’d’ve", "someonell": "someone’ll", "someones": "someone’s", "somethingd": "something’d", "somethingd’ve": "something’d’ve", "something’dve": "something’d’ve", "somethingll": "something’ll", "thats": "that’s", "thered": "there’d", "thered’ve": "there’d’ve", "there’dve": "there’d’ve", "therere": "there’re", "theres": "there’s", "theyd": "they’d", "theyd’ve": "they’d’ve", "they’dve": "they’d’ve", "theyll": "they’ll", "theyre": "they’re", "theyve": "they’ve", "twas": "’twas", "wasnt": "wasn’t", "wed’ve": "we’d’ve", "we’dve": "we’d’ve", "weve": "we've", "werent": "weren’t", "whatll": "what’ll", "whatre": "what’re", "whats": "what’s", "whatve": "what’ve", "whens": "when’s", "whered": "where’d", "wheres": "where's", "whereve": "where’ve", "whod": "who’d", "whod’ve": "who’d’ve", "who’dve": "who’d’ve", "wholl": "who’ll", "whos": "who’s", "whove": "who've", "whyll": "why’ll", "whyre": "why’re", "whys": "why’s", "wont": "won’t", "wouldve": "would’ve", "wouldnt": "wouldn’t", "wouldnt’ve": "wouldn’t’ve", "wouldn’tve": "wouldn’t’ve", "yall": "y’all", "yall’ll": "y’all’ll", "y’allll": "y’all’ll", "yall’d’ve": "y’all’d’ve", "y’alld’ve": "y’all’d’ve", "y’all’dve": "y’all’d’ve", "youd": "you’d", "youd’ve": "you’d’ve", "you’dve": "you’d’ve", "youll": "you’ll", "youre": "you’re", "youve": "you’ve"}

def processPunctuation(inText: str) -> str:
    outText = inText
    for p in punct:
        if (p + " " in inText or " " + p in inText) or (re.search(commaStrip, inText) is not None):
            outText = outText.replace(p, "")
        else:
            outText = outText.replace(p, " ")
    outText = periodStrip.sub("", outText, re.UNICODE)
    return outText


def processDigitArticle(inText: str) -> str:
    words = []
    for word in inText.lower().split():
        word = manualMap.get(word, word)
        if word in articles:
            continue
        words.append(contractions.get(word, word))
    return " ".join(words)


def normalize_answer(text: str) -> str:
    """The scorer's normalization of one answer."""
    return processDigitArticle(processPunctuation(str(text).replace("\n", " ").replace("\t", " ").strip()))


def consensus_index(answers: List[Dict]) -> Optional[int]:
    """Index of the plurality answer among those not marked ``answer_confidence: no`` (normalized;
    the earliest occurrence wins ties), or None if there is no such non-empty answer."""
    confident = [(i, normalize_answer(a.get("answer", ""))) for i, a in enumerate(answers)
                 if a.get("answer_confidence") != "no"]
    confident = [(i, n) for i, n in confident if n]
    if not confident:
        return None
    counts = Counter(n for _, n in confident)
    best = max(counts.values())
    return next(i for i, n in confident if counts[n] == best)


def demonstration_label(annotation: Dict) -> str:
    """Label a train item carries as a demonstration: its plurality answer (see ``consensus_index``);
    unanswerable items are labelled "unanswerable"."""
    answers = annotation.get("answers") or []
    if annotation.get("answerable") != 1 or not answers:
        return "unanswerable"
    i = consensus_index(answers)
    return answers[i if i is not None else 0]["answer"]
