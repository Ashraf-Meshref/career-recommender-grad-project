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

# Temperature > 1 softens an overconfident softmax (e.g. 100%/0%/0%) into a
# more realistic spread (e.g. 70%/20%/10%) without changing the ranking order.
# Tune this value: try 1.5, 2.0, 3.0, 4.0 and see what "feels" right.
SOFTMAX_TEMPERATURE = 2.5

def predict_career(skills_text, temperature=SOFTMAX_TEMPERATURE):
    cleaned = clean_skills(skills_text)
    vec     = tfidf.transform([cleaned]).toarray().astype(np.float32)
    tensor  = torch.tensor(vec).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits / temperature, dim=1).cpu().numpy()[0]
    ranked = sorted(
        [(le.classes_[i], float(probs[i])) for i in range(len(le.classes_))],
        key=lambda x: -x[1]
    )
    return ranked

# ── The 27 skills, mapped to easy, descriptive Yes/No questions ─────────────
# (skill_name, question_text)
QUESTIONS = [
    ("Web Development",
     "Imagine sitting down every day to build websites: designing pages, adding buttons, "
     "and making sure everything works when people click on it. Would you enjoy doing that as your job?"),

    ("Mobile App Development",
     "Imagine spending your day building phone apps: designing each screen, adding features, "
     "and testing them on a real phone. Would you enjoy doing that as your job?"),

    ("Artificial Intelligence (AI) and Machine Learning",
     "Imagine spending your day building smart systems like ChatGPT or Siri: teaching a computer "
     "to understand language, recognize patterns, or make decisions on its own. Would you enjoy doing that as your job?"),

    ("Problem Solving and Analysis",
     "Imagine spending your day looking at broken or confusing situations, breaking them into small "
     "parts, and working step by step until you find the real cause and fix it. Would you enjoy doing that as your job?"),

    ("Cybersecurity",
     "Imagine spending your day protecting computers and accounts from hackers: finding weak points, "
     "blocking attacks, and keeping data safe. Would you enjoy doing that as your job?"),

    ("Operating Systems and Networking",
     "Imagine spending your day working on how computers connect and talk to each other: setting up "
     "networks, fixing connection problems, and making sure data travels correctly. Would you enjoy doing that as your job?"),

    ("Database Development",
     "Imagine spending your day building and organizing huge digital storage systems, like the ones "
     "that hold millions of customer orders, so information can be found instantly. Would you enjoy doing that as your job?"),

    ("Data Analysis and Visualization",
     "Imagine spending your day looking at piles of numbers, like sales or survey results, and turning "
     "them into clear charts and graphs that explain what's happening. Would you enjoy doing that as your job?"),

    ("API Testing",
     "Imagine spending your day checking that different apps and systems are connecting to each other "
     "correctly, like making sure a food delivery app can read a restaurant's menu without errors. Would you enjoy doing that as your job?"),

    ("Performance Testing",
     "Imagine spending your day testing apps and websites to find out exactly why they are slow, then "
     "working to make them faster. Would you enjoy doing that as your job?"),

    ("Statistical Analysis",
     "Imagine spending your day using numbers and past data to calculate the chances of something "
     "happening, similar to how weather or sports predictions are made. Would you enjoy doing that as your job?"),

    ("Deep Learning",
     "Imagine spending your day training computer systems to recognize faces, understand speech, or "
     "spot patterns by studying thousands of examples, without writing exact step-by-step rules. Would you enjoy doing that as your job?"),

    ("Machine Learning",
     "Imagine spending your day building systems that learn and improve on their own by studying lots "
     "of examples, instead of you writing every instruction by hand. Would you enjoy doing that as your job?"),

    ("Data Engineering",
     "Imagine spending your day building the systems that move huge amounts of data automatically from "
     "one place to another inside a company, so it's ready for other people to use. Would you enjoy doing that as your job?"),

    ("Cloud Computing",
     "Imagine spending your day setting up and managing services like Netflix or Google Drive, which "
     "run on remote servers instead of one single computer. Would you enjoy doing that as your job?"),

    ("Blockchain",
     "Imagine spending your day building secure systems, like the ones behind Bitcoin, that record "
     "transactions without needing a bank in the middle. Would you enjoy doing that as your job?"),

    ("System Design",
     "Imagine spending your day planning, at a big-picture level, how all the different parts of a "
     "large software system should fit and work together before anyone starts building it. Would you enjoy doing that as your job?"),

    ("Project Management",
     "Imagine spending your day organizing people, schedules, and tasks, making sure everyone knows "
     "what to do so a project finishes on time. Would you enjoy doing that as your job?"),

    ("Game Development",
     "Imagine spending your day designing and building video games, from characters and levels to "
     "the rules of how the game works. Would you enjoy doing that as your job?"),

    ("Network Security",
     "Imagine spending your day defending a company's internet connections and servers from outside "
     "attacks, like guarding the doors and windows of a building. Would you enjoy doing that as your job?"),

    ("Graphic Design",
     "Imagine spending your day choosing colors, fonts, and layouts, and designing logos, posters, or "
     "other visuals to make things look attractive. Would you enjoy doing that as your job?"),

    ("UI/UX Knowledge",
     "Imagine spending your day redesigning confusing apps and websites to make them simple and "
     "pleasant for people to use. Would you enjoy doing that as your job?"),

    ("Internet of Things",
     "Imagine spending your day building or programming smart everyday devices, like smart watches, "
     "smart light bulbs, or smart thermostats, that connect to the internet. Would you enjoy doing that as your job?"),

    ("Big Data Technologies",
     "Imagine spending your day managing extremely large amounts of data, more than a normal computer "
     "can handle alone, using special tools built for companies like Amazon or Google. Would you enjoy doing that as your job?"),

    ("Image Processing",
     "Imagine spending your day building tools that can automatically blur backgrounds, detect faces, "
     "or improve photos, by working directly with the pixels in images. Would you enjoy doing that as your job?"),

    ("Feature Engineering",
     "Imagine spending your day cleaning and reshaping messy data so it becomes useful, similar to "
     "tidying a messy room before you can really use it. Would you enjoy doing that as your job?"),

    ("Software Quality Testing",
     "Imagine spending your day trying to break apps on purpose, before they are released, to find "
     "bugs and mistakes that other people missed. Would you enjoy doing that as your job?"),
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
    st.markdown(f"{question_text}  \nThis relates to the skill: {skill_name}")

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
        st.warning("You answered 'No' to everything, so there's nothing to match yet. "
                    "Try the quiz again and say Yes to at least a few things you enjoy.")
    else:
        skills_text = ", ".join(yes_skills)
        ranked = predict_career(skills_text)
        top3 = ranked[:3]

        st.subheader("🏆 Your Top 3 Career Matches")
        medals = ["🥇", "🥈", "🥉"]
        for medal, (career, conf) in zip(medals, top3):
            st.markdown(f"### {medal} {career}")
            st.progress(min(conf, 1.0))
            st.caption(f"Confidence: {conf:.1%}")
            st.markdown("")

    st.button("🔄 Retake the quiz", on_click=restart_quiz)
