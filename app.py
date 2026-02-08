import streamlit as st
import pandas as pd
import joblib

# ===== LOAD MODEL =====
model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="AI Student Grade Predictor",
    page_icon="🎓",
    layout="centered"
)

# ===== DARK UI STYLE =====
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#0f172a,#020617);
    color: white;
}

h1 {
    text-align: center;
    background: linear-gradient(90deg,#38bdf8,#818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Button */
.stButton>button {
    width: 100%;
    height: 3em;
    border-radius: 10px;
    font-size: 18px;
    font-weight: bold;
    color: white;
    background: linear-gradient(90deg,#6366f1,#22d3ee);
}

.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 15px #22d3ee;
}

/* Prediction box */
.pred {
    text-align:center;
    font-size:26px;
    font-weight:bold;
    padding:18px;
    border-radius:12px;
    background: rgba(0,0,0,0.4);
    color:#22d3ee;
    margin-top:20px;
}
</style>
""", unsafe_allow_html=True)

# ===== TITLE =====
st.title("🤖 AI Student Grade Predictor")

# ===== INPUT LAYOUT =====
col1, col2 = st.columns(2)

with col1:
    parent = st.selectbox(
        "🎓 Parent Education",
        ["some high school","high school","some college",
         "associate's degree","bachelor's degree","master's degree"]
    )

    reading = st.slider("📖 Reading Score",0,100,50)

    lunch = st.selectbox(
        "🍱 Lunch",
        ["standard","free/reduced"]
    )

with col2:
    writing = st.slider("✍️ Writing Score",0,100,50)

    prep = st.selectbox(
        "📝 Test Prep",
        ["none","completed"]
    )

# ===== PREDICTION =====
if st.button("🚀 Predict Math Score"):

    data = pd.DataFrame({
        "parental level of education":[parent],
        "reading score":[reading],
        "writing score":[writing],
        "lunch":[lunch],
        "test preparation course":[prep],
        "gender":["male"]
    })

    data = pd.get_dummies(data)
    data = data.reindex(columns=columns, fill_value=0)

    pred = model.predict(data)[0]

    st.markdown(
        f"<div class='pred'>Predicted Score: {round(pred,1)}</div>",
        unsafe_allow_html=True
    )

