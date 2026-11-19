"""
Batch validation: run the full image -> OCR -> evaluation pipeline on all 24 images and
compare against student_answers.csv.

student_answers.csv is used ONLY here, to (a) check OCR accuracy and (b) compare marks with
the benchmark. It is never an input to evaluation.

Usage:  python validate.py [dataset_dir] [output_dir]
"""
import os
import sys
from difflib import SequenceMatcher

import pandas as pd
from PIL import Image

import evaluator as ev

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(HERE), "MCA_Assignment_Evaluation_Dataset")
OUT = sys.argv[2] if len(sys.argv) > 2 else HERE

questions, rubric = ev.load_faculty_data(DATA)
truth = pd.read_csv(os.path.join(DATA, "student_answers.csv"), encoding="utf-8-sig")

rows = []
for _, t in truth.iterrows():
    img = Image.open(os.path.join(DATA, t["answer_image"]))
    fields, res = ev.evaluate_image(img, questions, rubric)
    ocr_sim = SequenceMatcher(None, fields.answer_text.lower(), t["student_answer_text_for_validation_only"].lower()).ratio()
    rows.append({
        "student_id": t["student_id"], "question_id": t["question_id"],
        "id_ok": fields.student_id == t["student_id"],
        "qid_ok": fields.question_id == t["question_id"],
        "ocr_answer_similarity": round(ocr_sim, 3),
        "ocr_answer": fields.answer_text,
        "system_marks": res.marks, "expected_marks": t["expected_marks_out_of_5"],
        "abs_error": abs(res.marks - t["expected_marks_out_of_5"]),
        "system_category": res.category, "expected_category": t["expected_category"],
        "category_ok": res.category == t["expected_category"],
        "similarity": round(res.similarity, 3),
        "covered": "; ".join(res.covered), "missing": "; ".join(res.missing),
        "feedback": res.feedback, "flags": " | ".join(res.flags),
    })

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "validation_results.csv"), index=False, encoding="utf-8-sig")

print(df[["student_id", "question_id", "system_marks", "expected_marks", "system_category",
          "expected_category", "ocr_answer_similarity"]].to_string(index=False))
print()
print(f"Student ID read correctly   : {df.id_ok.sum()}/{len(df)}")
print(f"Question ID read correctly  : {df.qid_ok.sum()}/{len(df)}")
print(f"Mean OCR answer similarity  : {df.ocr_answer_similarity.mean():.3f}")
print(f"Mean absolute mark error    : {df.abs_error.mean():.2f} marks (out of 5)")
print(f"Within +/-1 mark of benchmark: {(df.abs_error <= 1).sum()}/{len(df)}")
print(f"Category matches benchmark  : {df.category_ok.sum()}/{len(df)}")
print(f"Pearson r (system vs expected): {df.system_marks.corr(df.expected_marks):.3f}")
