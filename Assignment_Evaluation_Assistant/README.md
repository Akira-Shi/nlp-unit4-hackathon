# Intelligent Student Assignment Evaluation Assistant

Reads a student's answer-sheet **image**, extracts the text with OCR, finds the matching
faculty question, compares the answer with the reference answer and the marking rubric,
and produces preliminary marks plus feedback. Marks are preliminary; faculty make the final call.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit app (Steps 1-8 of the brief, "Evaluate another image" button) |
| `evaluator.py` | The engine: OCR, parsing, NLP preprocessing, similarity, rubric scoring, feedback |
| `validate.py` | Runs all 24 images through the pipeline and compares with `student_answers.csv` |
| `validation_results.csv` | Output of `validate.py` (per-image marks, OCR check, feedback) |
| `requirements.txt` | Python dependencies |

## Run it

```
pip install -r requirements.txt          # also install the Tesseract engine (see requirements.txt)
streamlit run app.py                     # then upload any image from answer_images/
python validate.py                       # batch check against the benchmark
```
The app expects `MCA_Assignment_Evaluation_Dataset/` one level above this folder
(editable in the sidebar).

## How it works

1. **OCR** (`run_ocr`, `parse_sheet`). Tesseract on a grayscale, 2x upscaled, Otsu-thresholded
   image; a second pass on the raw image is used if the Student ID looks corrupted. Regex pulls
   out Student ID, question number, question text and answer. Safe OCR repairs: `0/O/Q` confusion
   inside IDs, `Al` read for `AI`. EasyOCR is an automatic fallback.
   `student_answers.csv` is **never** an input to evaluation.
2. **Question lookup.** The printed Q-number is matched to `questions.csv`. If it cannot be read,
   the OCR'd question text is fuzzy-matched instead, and the report flags this.
3. **Preprocessing.** Lowercase, strip punctuation, tokenize, remove stop words, Porter stemming
   (shown step by step in the app).
4. **Comparison.** TF-IDF cosine similarity against the reference answer, computed after
   synonym-canonicalisation and acronym expansion (NLP, AI, ML, CNN). Shown for information only.
5. **Rubric scoring (this decides the marks).** Each rubric line is a concept; `a / b` means
   alternatives. A concept is
   - **covered** (full mark) when all its content words appear in the answer, synonyms allowed
     (e.g. *divides* = *splits*, *picture* = *image*, *finds* = *identifies*);
   - **partly covered** (half mark) when a multi-word concept is mostly there (>= 2 words, >= 60%);
   - **missing** otherwise. A single stray word from a multi-word concept earns nothing.
   Negations such as *unlabelled* do not match *labelled* (different stem).
6. **Relevance gate.** If the answer shares no content word with the question and similarity is low,
   marks are set to 0 and flagged, so generic words (*words*, *computers*) cannot earn marks in an
   off-topic answer.
7. **Feedback.** Lists concepts covered, partly covered and missing, with a suggestion.
8. **Report.** Student ID, question, max vs preliminary marks, category, concepts, feedback, flags.

## Validation results (24 images, `python validate.py`)

| Check | Result |
|---|---|
| Question number read from image | 24 / 24 |
| OCR answer vs `student_answers.csv` text (similarity) | 1.00 on all 24 |
| Student ID read correctly | 24 / 24 |
| Marks within +/-1 of benchmark | 24 / 24 |
| Category (Relevant / Partial / Irrelevant) matches | 24 / 24 |
| Mean absolute mark error | about 0.2 marks out of 5 |

The five images that differ from the benchmark (by 0.5-1 mark) are all under-scored. The benchmark
is described in the dataset README as illustrative; for example it awards credit for "knows the
answers" in *A model learns from examples with known answers* (a paraphrase of *labelled data*) which
a word-level matcher does not recognise. Faculty review catches these.

## Limitations (honest list)

- Concept matching is lexical plus a hand-built synonym list. Paraphrases outside the list
  (e.g. *known answers* for *labelled data*) are missed. A sentence-embedding model
  (e.g. `sentence-transformers`) is the natural upgrade for semantic partial credit.
- The synonym list was written with this 8-question set in mind; new questions may need new entries.
- Printed text only. Handwriting would need a handwriting-capable OCR model.
- Half-marks are awarded in 0.5 steps; the benchmark uses whole marks.
