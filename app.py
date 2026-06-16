# app_streamlit2.py  ── Career Path Recommender (Yes/No Quiz version)
# Run: streamlit run app_streamlit2.py
#
# Instead of asking the user to pick skills by name, this version asks one
# plain-language Yes/No question per skill, then feeds the "Yes" skills into
# the same trained model used by app_streamlit.py and shows only the Top 3
# recommended careers at the end.

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import joblib
import json
import re
import pandas as pd
import plotly.express as px

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Career Path Recommender — Quiz",
    page_icon="🧭",
    layout="centered"
)

# ── Model definition (must match training) ──────────────────────────────────
class CareerMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(1024, 512),      nn.BatchNorm1d(512),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512,  256),      nn.BatchNorm1d(256),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256,  128),                            nn.ReLU(),
            nn.Linear(128,  num_classes)
        )
    def forward(self, x):
        return self.net(x)

# ── Load artifacts (cached) ──────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    tfidf = joblib.load("artifacts/tfidf_vectorizer.pkl")
    le    = joblib.load("artifacts/label_encoder.pkl")
    with open("artifacts/model_meta.json") as f:
        meta = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = CareerMLP(meta["input_dim"], meta["num_classes"]).to(device)
    model.load_state_dict(torch.load("artifacts/best_model.pt", map_location=device))
    model.eval()
    return tfidf, le, model, meta, device

tfidf, le, model, meta, device = load_artifacts()

def clean_skills(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9,\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict_career(skills_text):
    cleaned = clean_skills(skills_text)
    vec     = tfidf.transform([cleaned]).toarray().astype(np.float32)
    tensor  = torch.tensor(vec).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).cpu().numpy()[0]
    ranked = sorted(
        [(le.classes_[i], float(probs[i])) for i in range(len(le.classes_))],
        key=lambda x: -x[1]
    )
    return ranked

# ── The 27 skills, mapped to plain-language Yes/No questions ────────────────
# (skill_name, question_text)
QUESTIONS = [
    ("Web Development",
     "When you visit a page online, are you curious how its look, buttons, and pages were actually made?"),
    ("Mobile App Development",
     "Do you ever look at an app on your phone and wonder how it was built?"),
    ("Artificial Intelligence (AI) and Machine Learning",
     "Are you fascinated by computers that can make decisions or 'think' on their own, like chatbots or self-driving cars?"),
    ("Problem Solving and Analysis",
     "When something is broken or confusing, do you enjoy digging in step by step until you figure out why?"),
    ("Cybersecurity",
     "Do you find it interesting how hackers break into systems, and how people defend against them?"),
    ("Operating Systems and Networking",
     "Are you curious how your phone connects to WiFi, or how computers talk to each other over the internet?"),
    ("Database Development",
     "Do you like the idea of organizing huge amounts of information so it's easy to find later (like a giant digital filing cabinet)?"),
    ("Data Analysis and Visualization",
     "Do you enjoy looking at numbers or spreadsheets and turning them into charts that tell a story?"),
    ("API Testing",
     "Do you enjoy checking whether two separate apps or systems actually work correctly when connected together?"),
    ("Performance Testing",
     "Are you the type of person who gets annoyed when an app is slow, and wants to know exactly why?"),
    ("Statistical Analysis",
     "Do you enjoy working with probabilities and numbers, like predicting outcomes or analyzing surveys?"),
    ("Deep Learning",
     "Are you curious about how a computer can learn to recognize a face or voice just from seeing lots of examples?"),
    ("Machine Learning",
     "Would you find it exciting to teach a computer to improve at a task just by giving it more data, instead of writing exact rules?"),
    ("Data Engineering",
     "Do you like the idea of building the 'pipes' that move information from one place to another automatically?"),
    ("Cloud Computing",
     "Have you heard of things like Google Drive or Netflix running on remote servers, and are you curious how that works?"),
    ("Blockchain",
     "Are you curious about how cryptocurrencies like Bitcoin keep records securely without one central authority?"),
    ("System Design",
     "Do you enjoy planning things out on a big-picture level, like how all the pieces of a large project should fit together?"),
    ("Project Management",
     "Do you enjoy organizing people, deadlines, and tasks to make sure a project gets finished on time?"),
    ("Game Development",
     "Have you ever played a video game and wished you could build your own?"),
    ("Network Security",
     "Are you interested in protecting a company's internet connections and servers from outside attacks?"),
    ("Graphic Design",
     "Do you enjoy making things look visually appealing, like posters, logos, or color schemes?"),
    ("UI/UX Knowledge",
     "Do you get frustrated when an app or website is confusing to use, and think about how it could be simpler?"),
    ("Internet of Things",
     "Are you interested in smart gadgets, like smart watches, smart thermostats, or smart light bulbs?"),
    ("Big Data Technologies",
     "Are you interested in how companies like Amazon or Google handle and store enormous amounts of data?"),
    ("Image Processing",
     "Are you curious how apps can automatically blur backgrounds, detect faces, or enhance photos?"),
    ("Feature Engineering",
     "Do you enjoy cleaning up messy information and reshaping it so it's more useful before analyzing it?"),
    ("Software Quality Testing",
     "Do you have a knack for spotting mistakes or things that don't work right before anyone else notices?"),
]
TOTAL_Q = len(QUESTIONS)

# ── Session state ─────────────────────────────────────────────────────────
if "q_index" not in st.session_state:
    st.session_state.q_index = 0
if "answers" not in st.session_state:
    st.session_state.answers = {}   # skill_name -> True/False
if "finished" not in st.session_state:
    st.session_state.finished = False

def answer_question(skill_name, value):
    st.session_state.answers[skill_name] = value
    st.session_state.q_index += 1
    if st.session_state.q_index >= TOTAL_Q:
        st.session_state.finished = True

def restart_quiz():
    st.session_state.q_index = 0
    st.session_state.answers = {}
    st.session_state.finished = False

def go_back():
    if st.session_state.q_index > 0:
        st.session_state.q_index -= 1
        st.session_state.finished = False

# ── UI ──────────────────────────────────────────────────────────────────────
st.title("🧭 Career Path Recommender")
st.caption("Not sure what your skills are called? Just answer Yes or No.")

if not st.session_state.finished:
    idx = st.session_state.q_index
    skill_name, question_text = QUESTIONS[idx]

    st.progress(idx / TOTAL_Q)
    st.markdown(f"**Question {idx + 1} of {TOTAL_Q}**")
    st.subheader(question_text)

    col1, col2 = st.columns(2)
    with col1:
        st.button("✅ Yes", use_container_width=True, key=f"yes_{idx}",
                   on_click=answer_question, args=(skill_name, True))
    with col2:
        st.button("❌ No", use_container_width=True, key=f"no_{idx}",
                   on_click=answer_question, args=(skill_name, False))

    if idx > 0:
        st.button("← Back", on_click=go_back)

else:
    st.success("Quiz complete! Here are your top career matches.")

    yes_skills = [skill for skill, val in st.session_state.answers.items() if val]

    if not yes_skills:
        st.warning("You answered 'No' to everything. Try the quiz again and select at least a few areas that spark your interest.")
    else:
        skills_text = ", ".join(yes_skills)
        ranked = predict_career(skills_text)
        top3 = ranked[:3]

        st.subheader("🏆 Your Top 3 Recommended Careers")
        
        # Displaying only the Top 3 with confidence
        for i, (career, conf) in enumerate(top3):
            st.markdown(f"### {i+1}. {career}")
            st.progress(min(conf, 1.0))
            st.write(f"Match Confidence: **{conf:.1%}**")
            st.write("---")

    st.button("🔄 Retake the quiz", on_click=restart_quiz)