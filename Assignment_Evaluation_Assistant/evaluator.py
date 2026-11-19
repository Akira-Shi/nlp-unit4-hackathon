"""
Core engine for the Intelligent Student Assignment Evaluation Assistant.

Pipeline (mirrors the hackathon brief):
  image -> OCR -> field parsing -> question lookup -> NLP preprocessing
        -> TF-IDF similarity -> rubric concept matching -> marks -> feedback

Nothing here reads student_answers.csv: the answer always comes from the image.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_stem = PorterStemmer().stem

# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_faculty_data(dataset_dir: str):
    """Load faculty-provided questions and rubric (utf-8-sig strips the BOM)."""
    q = pd.read_csv(os.path.join(dataset_dir, "questions.csv"), encoding="utf-8-sig")
    r = pd.read_csv(os.path.join(dataset_dir, "marking_rubric.csv"), encoding="utf-8-sig")
    q["question_id"] = q["question_id"].str.strip().str.upper()
    r["question_id"] = r["question_id"].str.strip().str.upper()
    return q, r


# --------------------------------------------------------------------------
# OCR
# --------------------------------------------------------------------------

def _preprocess_for_ocr(pil_img):
    """Grayscale -> 2x upscale -> Otsu threshold. Helps Tesseract on photos/scans."""
    import cv2

    arr = np.array(pil_img.convert("L"))
    arr = cv2.resize(arr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, arr = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return arr


ID_PATTERN = re.compile(r"^[A-Z]{2,4}\d{3,4}$")


def _fix_id(token: str) -> str:
    """Repair common OCR confusions inside an ID like MCA002 (letters first, then digits)."""
    token = re.sub(r"[^A-Za-z0-9]", "", token).upper()
    m = re.match(r"([A-Z]{3})(.*)", token)
    if not m:
        return token
    digits = m.group(2).translate(str.maketrans("OQDIL|SB", "00011158"))
    return m.group(1) + digits


def _id_from_text(text: str, repair: bool = True) -> str:
    m = re.search(r"student\s*id\s*[:\-]?\s*(\S+)", text, re.I)
    if not m:
        return ""
    return _fix_id(m.group(1)) if repair else re.sub(r"[^A-Za-z0-9]", "", m.group(1)).upper()


def _clean_ocr(text: str) -> str:
    """Fix OCR confusions that are safe in this domain (capital-I read as lowercase-l)."""
    return re.sub(r"\bAl\b", "AI", text)


def run_ocr(pil_img) -> tuple[str, str]:
    """Return (text, engine_name). Tesseract first; EasyOCR if Tesseract is missing.
    Two passes (binarised 2x image, then the raw image); the pass whose Student ID
    looks like a valid ID wins, so a single noisy pass does not corrupt the result."""
    try:
        import pytesseract

        best, repaired_ok = "", ""
        for variant in (_preprocess_for_ocr(pil_img), np.array(pil_img.convert("L"))):
            text = pytesseract.image_to_string(variant, config="--psm 6")
            if not best and text.strip():
                best = text
            # a pass whose ID is valid as read wins outright; an ID that only becomes
            # valid after character repair is kept as a fallback
            if ID_PATTERN.match(_id_from_text(text, repair=False)):
                return _clean_ocr(text), "Tesseract"
            if not repaired_ok and ID_PATTERN.match(_id_from_text(text)):
                repaired_ok = text
        if repaired_ok or best:
            return _clean_ocr(repaired_ok or best), "Tesseract"
    except Exception:
        pass
    try:
        import easyocr

        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        lines = reader.readtext(np.array(pil_img.convert("RGB")), detail=0, paragraph=False)
        return "\n".join(lines), "EasyOCR"
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "No OCR engine available. Install Tesseract (+ pytesseract) or EasyOCR."
        ) from e


@dataclass
class SheetFields:
    student_id: str = ""
    question_id: str = ""
    question_text: str = ""
    answer_text: str = ""
    raw_text: str = ""
    engine: str = ""


def parse_sheet(raw: str, engine: str = "") -> SheetFields:
    """Pull Student ID, Question number, question text and answer out of OCR text."""
    f = SheetFields(raw_text=raw, engine=engine)
    flat = raw.replace("\r", "")

    f.student_id = _id_from_text(flat)

    m = re.search(r"question\s*[:\-]?\s*(Q\s?\d+)", flat, re.I)
    if m:
        f.question_id = re.sub(r"\s", "", m.group(1)).upper()

    # NB: the page title reads "STUDENT ANSWER SHEET" (no colon) - only the labelled field splits.
    parts = re.split(r"student\s*answer\s*:", flat, maxsplit=1, flags=re.I)
    head = parts[0]
    tail = parts[1] if len(parts) > 1 else ""

    # question text = everything after the LAST "Question:" label in the head
    chunks = re.split(r"question\s*[:\-]?", head, flags=re.I)
    cand = chunks[-1] if chunks else ""
    if re.fullmatch(r"\s*Q\s?\d+\s*", cand, re.I):  # last label was the "Question: Q5" header
        cand = ""
    f.question_text = " ".join(cand.split())

    tail = re.split(r"end\s+of\s+answer", tail, flags=re.I)[0]
    f.answer_text = " ".join(tail.split())
    return f


def resolve_question_id(fields: SheetFields, questions: pd.DataFrame) -> tuple[str, str]:
    """Use the printed Q-number if it exists in questions.csv; otherwise fall back to
    fuzzy matching of the OCR'd question text. Returns (question_id, how)."""
    ids = set(questions["question_id"])
    if fields.question_id in ids:
        return fields.question_id, "question number read from image"
    if fields.question_text:
        best, score = "", 0.0
        for _, row in questions.iterrows():
            s = SequenceMatcher(None, fields.question_text.lower(), row["question"].lower()).ratio()
            if s > score:
                best, score = row["question_id"], s
        if score >= 0.6:
            return best, f"matched by question text (similarity {score:.2f})"
    return "", "not found"


# --------------------------------------------------------------------------
# NLP preprocessing
# --------------------------------------------------------------------------

STOPWORDS = set("""a an the is are was were be been being am of to in on at by for from with as that this
these those it its it's and or but so such than then there their they them he she we you i do does did
done has have had having can could may might will would shall should into onto over under about also
very just which who whom whose what when where why how not no nor if else while using use used uses
helps help helping like etc one ones some any each other another more most much many""".split())

# Acronyms students commonly use. Expansion tokens are *added* to the answer.
ACRONYMS = {
    "nlp": ["natural", "language", "processing"],
    "ai": ["artificial", "intelligence"],
    "ml": ["machine", "learning"],
    "cnn": ["convolutional", "neural", "network"],
}

# Groups of interchangeable words. Every member maps to the first stem of the group.
SYNONYM_GROUPS = [
    ["understand", "comprehend", "grasp", "understanding"],
    ["process", "handle", "manipulate", "analyze", "analyse", "work"],
    ["computer", "computing", "system", "software"],
    ["split", "divide", "break", "separate", "partition", "chop", "cut", "segment"],
    ["unit", "piece", "part", "chunk", "token", "element", "portion"],
    ["text", "paragraph", "document", "string", "corpus", "sentence"],
    ["identify", "find", "detect", "determine", "classify", "recognize", "recognise", "extract"],
    ["correct", "fix", "handle", "repair", "resolve", "treat", "impute", "remove", "clean"],
    ["incorrect", "wrong", "inconsistent", "inconsistency", "erroneous", "error", "invalid"],
    ["duplicate", "duplicates", "repeated", "redundant"],
    ["missing", "null", "empty", "absent", "blank"],
    ["filter", "smooth", "blur", "sharpen"],
    ["resize", "rescale", "scale", "dimension"],
    ["normalize", "normalise", "standardize", "standardise", "normalization", "standardization"],
    ["improve", "enhance", "better", "boost", "increase"],
    ["quality", "clarity"],
    ["noise", "denoise", "denoising"],
    ["emotion", "feeling", "sentiment", "mood"],
    ["opinion", "view", "attitude", "polarity"],
    ["label", "labelled", "labeled", "annotated", "tagged", "known"],
    ["predict", "forecast", "estimate", "infer"],
    ["output", "result", "prediction", "target"],
    ["unseen", "new", "future", "test", "fresh"],
    ["learn", "train", "fit"],
    ["region", "area", "part", "section", "zone"],
    ["object", "entity", "item"],
    ["interest", "interesting", "important", "relevant"],
    ["feature", "pattern", "characteristic"],
    ["image", "picture", "photo", "visual"],
    ["architecture", "network", "neural", "model"],
    ["technique", "method", "approach", "algorithm", "task"],
    ["meaningful", "distinct", "different", "homogeneous", "similar"],
]
_CANON: dict[str, str] = {}
for grp in SYNONYM_GROUPS:
    head = _stem(grp[0])
    for w in grp:
        _CANON.setdefault(_stem(w), head)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation/special characters, collapse whitespace."""
    return " ".join(re.sub(r"[^a-z0-9\s]", " ", text.lower()).split())


def content_tokens(text: str, expand_acronyms: bool = False) -> list[str]:
    """Tokenize -> drop stop words -> stem. Optionally add acronym expansions."""
    toks = tokenize(text)
    if expand_acronyms:
        extra = []
        for t in toks:
            extra += ACRONYMS.get(t, [])
        toks = toks + extra
    return [_stem(t) for t in toks if t not in STOPWORDS and len(t) > 1]


def canon(stem: str) -> str:
    return _CANON.get(stem, stem)


def answer_concept_set(text: str) -> tuple[set, set]:
    """Return (raw_stems, canonical_stems) for an answer. Reverse acronym expansion
    (e.g. 'natural language processing' -> 'nlp') is applied too."""
    toks = tokenize(text)
    # words with an 'un'/'non' prefix are *negations* of the base word (unlabelled != labelled)
    raw = set()
    for t in toks:
        if t in STOPWORDS or len(t) < 2:
            continue
        raw.add(_stem(t))
    for ac, exp in ACRONYMS.items():
        if ac in toks:
            raw.update(_stem(x) for x in exp)
        if all(_stem(x) in raw for x in exp):
            raw.add(ac)
    return raw, {canon(s) for s in raw}


@dataclass
class Preprocessed:
    cleaned: str
    tokens: list
    without_stopwords: list
    stems: list


def preprocess(text: str) -> Preprocessed:
    cleaned = clean_text(text)
    toks = tokenize(cleaned)
    no_stop = [t for t in toks if t not in STOPWORDS]
    return Preprocessed(cleaned, toks, no_stop, [_stem(t) for t in no_stop])


# --------------------------------------------------------------------------
# Comparison + rubric
# --------------------------------------------------------------------------

def tfidf_similarity(student: str, reference: str) -> float:
    """Cosine similarity of TF-IDF vectors (on stemmed, canonicalised, acronym-expanded text)."""
    def norm(t):
        return " ".join(canon(s) for s in content_tokens(t, expand_acronyms=True))

    a, b = norm(student), norm(reference)
    if not a.strip() or not b.strip():
        return 0.0
    vec = TfidfVectorizer().fit([a, b])
    m = vec.transform([a, b])
    return float(cosine_similarity(m[0], m[1])[0, 0])


@dataclass
class ConceptResult:
    concept: str
    max_marks: float
    awarded: float
    status: str          # "covered" | "partial" | "missing"
    evidence: str = ""   # which alternative / words matched


def _match_alternative(alt: str, raw: set, canon_set: set) -> tuple[float, str]:
    """Fraction of an alternative's content words found in the answer."""
    toks = [_stem(t) for t in tokenize(alt) if t not in STOPWORDS and len(t) > 1]
    if not toks:
        return 0.0, ""
    hit = []
    for t in toks:
        if t in raw or canon(t) in canon_set:
            hit.append(t)
    return len(hit) / len(toks), " + ".join(hit)


def score_rubric(student_text: str, rubric_rows: pd.DataFrame) -> list[ConceptResult]:
    """Concept-by-concept scoring.
       * Rubric cells like 'architecture / neural network' are alternatives (any one works).
       * Full marks when every word of the best alternative is present (synonym-aware).
       * Half marks when a multi-word concept is only partly present (>=2 words and >=60%).
       * A single stray word out of a multi-word concept earns nothing."""
    raw, canon_set = answer_concept_set(student_text)
    out = []
    for _, row in rubric_rows.iterrows():
        concept, marks = row["concept_or_expected_point"], float(row["marks"])
        best_frac, best_ev, best_n = 0.0, "", 0
        for alt in [a.strip() for a in concept.split("/") if a.strip()]:
            frac, ev = _match_alternative(alt, raw, canon_set)
            n_hit = len(ev.split(" + ")) if ev else 0
            if frac > best_frac or (frac == best_frac and n_hit > best_n):
                best_frac, best_ev, best_n = frac, ev, n_hit
        if best_frac >= 0.999:
            out.append(ConceptResult(concept, marks, marks, "covered", best_ev))
        elif best_frac >= 0.6 and best_n >= 2:
            out.append(ConceptResult(concept, marks, marks / 2, "partial", best_ev))
        else:
            out.append(ConceptResult(concept, marks, 0.0, "missing", best_ev))
    return out


def topic_anchor(student_text: str, question_text: str) -> bool:
    """Does the answer share at least one content word (synonym/acronym aware) with the
    question? A zero overlap means the answer is about something else entirely."""
    _, ans = answer_concept_set(student_text)
    q = {canon(s) for s in content_tokens(question_text, expand_acronyms=True)}
    # 'what is X' boilerplate
    q -= {canon(_stem(w)) for w in ["what", "define", "explain", "describe"]}
    return bool(ans & q)


# --------------------------------------------------------------------------
# Feedback + full evaluation
# --------------------------------------------------------------------------

@dataclass
class Evaluation:
    student_id: str
    question_id: str
    question: str
    reference_answer: str
    max_marks: float
    marks: float
    similarity: float
    category: str
    concepts: list = field(default_factory=list)
    covered: list = field(default_factory=list)
    partial: list = field(default_factory=list)
    missing: list = field(default_factory=list)
    feedback: str = ""
    flags: list = field(default_factory=list)
    preprocessed: Preprocessed | None = None


def _pretty(c: str) -> str:
    return c.split("/")[0].strip() if "/" in c and len(c.split("/")[0].split()) > 1 else c.replace(" / ", " or ")


def build_feedback(question: str, covered, partial, missing, off_topic: bool, marks, max_marks) -> str:
    if off_topic:
        return ("Your answer does not address the question asked. Re-read the question and "
                f"write specifically about: {question.rstrip('?')}.")
    cov = ", ".join(_pretty(c) for c in covered) or "none of the expected ideas"
    msg = f"Your answer correctly covers: {cov}."
    if partial:
        msg += " Partly covered (be more precise): " + ", ".join(_pretty(c) for c in partial) + "."
    if missing:
        msg += " To improve, also mention: " + ", ".join(_pretty(c) for c in missing) + "."
    else:
        msg += " This is a complete answer."
    return msg


def evaluate(fields: SheetFields, questions: pd.DataFrame, rubric: pd.DataFrame) -> Evaluation:
    qid, how = resolve_question_id(fields, questions)
    flags = []
    if not qid:
        return Evaluation(fields.student_id, fields.question_id or "?", fields.question_text, "", 0, 0, 0.0,
                          "Unresolved", feedback="Question could not be identified - faculty must review.",
                          flags=["Question ID not found in questions.csv"])
    if how != "question number read from image":
        flags.append(f"Question {how}")

    qrow = questions[questions["question_id"] == qid].iloc[0]
    rub = rubric[rubric["question_id"] == qid]
    max_marks = float(qrow["maximum_marks"])
    ans = fields.answer_text

    pre = preprocess(ans)
    sim = tfidf_similarity(ans, qrow["reference_answer"])
    concepts = score_rubric(ans, rub)
    marks = sum(c.awarded for c in concepts)

    # Relevance gate: an answer with no connection to the question and weak similarity
    # must not collect marks from generic words that happen to be in the rubric.
    off_topic = (not topic_anchor(ans, qrow["question"])) and sim < 0.20
    if off_topic and marks > 0:
        flags.append(f"Off-topic answer: {marks:g} rubric mark(s) from generic words were withheld")
        marks = 0.0
        concepts = [ConceptResult(c.concept, c.max_marks, 0.0, "missing", c.evidence) for c in concepts]
    if not ans.strip():
        flags.append("No answer text was extracted")
    if fields.question_text:
        qs = SequenceMatcher(None, fields.question_text.lower(), qrow["question"].lower()).ratio()
        if qs < 0.8:
            flags.append(f"OCR'd question text differs from questions.csv (similarity {qs:.2f}) - check OCR quality")

    marks = min(marks, max_marks)
    ratio = marks / max_marks if max_marks else 0
    category = "Relevant" if ratio >= 0.8 else "Partially relevant" if ratio >= 0.4 else "Irrelevant"
    if off_topic:
        category = "Irrelevant"

    covered = [c.concept for c in concepts if c.status == "covered"]
    partial = [c.concept for c in concepts if c.status == "partial"]
    missing = [c.concept for c in concepts if c.status == "missing"]
    fb = build_feedback(qrow["question"], covered, partial, missing, off_topic, marks, max_marks)

    return Evaluation(fields.student_id, qid, qrow["question"], qrow["reference_answer"], max_marks, marks,
                      sim, category, concepts, covered, partial, missing, fb, flags, pre)


def evaluate_image(pil_img, questions, rubric) -> tuple[SheetFields, Evaluation]:
    text, engine = run_ocr(pil_img)
    fields = parse_sheet(text, engine)
    return fields, evaluate(fields, questions, rubric)
