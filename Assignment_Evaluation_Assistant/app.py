"""
Intelligent Student Assignment Evaluation Assistant - Streamlit app.
Run:  streamlit run app.py
"""
import os

import pandas as pd
import streamlit as st
from PIL import Image

import evaluator as ev

st.set_page_config(page_title="Assignment Evaluation Assistant", page_icon="📝", layout="wide")

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(os.path.dirname(HERE), "MCA_Assignment_Evaluation_Dataset")

# ---------------- sidebar: faculty data ----------------
st.sidebar.header("Faculty data")
data_dir = st.sidebar.text_input("Dataset folder", DEFAULT_DATA)
try:
    questions, rubric = ev.load_faculty_data(data_dir)
    st.sidebar.success(f"Loaded {len(questions)} questions, {len(rubric)} rubric points")
except Exception as e:
    st.sidebar.error(f"Could not load questions.csv / marking_rubric.csv: {e}")
    st.stop()

with st.sidebar.expander("View questions"):
    st.dataframe(questions[["question_id", "question", "maximum_marks"]], hide_index=True)

st.sidebar.caption("Marks are PRELIMINARY and must be reviewed by faculty before finalising.")

# "Evaluate another image" = bump the uploader key to reset it
if "run" not in st.session_state:
    st.session_state.run = 0

st.title("📝 Intelligent Student Assignment Evaluation Assistant")
st.caption("OCR → question lookup → NLP comparison → rubric marking → feedback. Final marks stay with faculty.")

# ---------------- Step 1: upload ----------------
st.header("Step 1 - Upload answer sheet")
up = st.file_uploader("Upload a student answer-sheet image", type=["png", "jpg", "jpeg"],
                      key=f"up{st.session_state.run}")
if up is None:
    st.info("Upload an image from `answer_images/` to begin.")
    st.stop()

img = Image.open(up)
c1, c2 = st.columns([3, 2])
c1.image(img, caption=up.name, use_container_width=True)

with st.spinner("Running OCR..."):
    text, engine = ev.run_ocr(img)
fields = ev.parse_sheet(text, engine)
result = ev.evaluate(fields, questions, rubric)

with c2:
    st.subheader("Identified details")
    st.metric("Student ID", fields.student_id or "not found")
    st.metric("Question number (from image)", fields.question_id or "not found")

# ---------------- Step 2: OCR ----------------
st.header(f"Step 2 - Extracted text ({engine})")
a, b = st.columns(2)
a.text_area("Question (OCR)", fields.question_text, height=80)
b.text_area("Student answer (OCR)", fields.answer_text, height=80)
with st.expander("Full raw OCR output"):
    st.code(fields.raw_text)

# ---------------- Step 3: question lookup ----------------
st.header("Step 3 - Identified question")
if result.category == "Unresolved":
    st.error("Question could not be matched to questions.csv. Faculty must review manually.")
    st.stop()
st.write(f"**{result.question_id}** - {result.question}")
st.write(f"**Reference answer:** {result.reference_answer}")
st.write(f"**Maximum marks:** {result.max_marks:g}")

# ---------------- Step 4: preprocessing ----------------
st.header("Step 4 - Preprocessed answer")
p = result.preprocessed
st.write("**Cleaned text:**", p.cleaned)
st.write("**Tokens:**", p.tokens)
st.write("**After stop-word removal:**", p.without_stopwords)
st.write("**Stems (important words):**", p.stems)

# ---------------- Step 5: comparison ----------------
st.header("Step 5 - Comparison with reference answer")
st.progress(min(result.similarity, 1.0), text=f"TF-IDF cosine similarity: {result.similarity:.2f}")
st.caption("Similarity is shown for information only - marks come from rubric concepts, "
           "because similarity alone does not establish correctness.")

# ---------------- Step 6: rubric ----------------
st.header("Step 6 - Rubric evaluation")
icon = {"covered": "✅ Yes", "partial": "🟡 Partly", "missing": "❌ No"}
tbl = pd.DataFrame([{
    "Expected concept": c.concept, "Found in answer?": icon[c.status],
    "Matched words": c.evidence, "Marks": f"{c.awarded:g} / {c.max_marks:g}"} for c in result.concepts])
st.dataframe(tbl, hide_index=True, use_container_width=True)

# ---------------- Step 7-8: feedback + report ----------------
st.header("Step 7 & 8 - Evaluation report")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Student ID", result.student_id)
m2.metric("Question", result.question_id)
m3.metric("Preliminary marks", f"{result.marks:g} / {result.max_marks:g}")
m4.metric("Category", result.category)

g, mi = st.columns(2)
with g:
    st.subheader("Concepts covered")
    for c in result.covered:
        st.write("✅", c)
    for c in result.partial:
        st.write("🟡", c, "(partly)")
    if not result.covered and not result.partial:
        st.write("None")
with mi:
    st.subheader("Concepts missing")
    for c in result.missing:
        st.write("❌", c)
    if not result.missing:
        st.write("None")

st.subheader("Feedback")
st.info(result.feedback)
for fl in result.flags:
    st.warning(fl)
st.caption("⚠️ Preliminary marks - to be reviewed and confirmed by faculty.")

if st.button("Evaluate another image"):
    st.session_state.run += 1
    st.rerun()
