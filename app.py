import streamlit as st
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torch.nn.functional import softmax
import torch
import base64
from pathlib import Path

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="🎬 Movie Review Sentiment Analyzer",
    page_icon="🎥",
    layout="centered",
)

# --- ENCODE LOCAL IMAGE ---
def get_base64_of_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

image_path = r"C:\Users\DELL\Downloads\arrangement-movie-elements-black-background.jpg"
image_base64 = get_base64_of_image(image_path)

# --- CSS STYLING: 3D CARD + ENHANCED UI ---
background_style = f"""
<style>
    /* Background (unchanged) */
    .stApp {{
        background-image: url("data:image/jpg;base64,{image_base64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    /* Main 3D Floating Card */
    .card-container {{
        perspective: 1000px;
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
    }}

    .card {{
        background: rgba(14, 17, 23, 0.92);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 24px;
        padding: 3rem 2.5rem;
        max-width: 820px;
        width: 100%;
        border: 1px solid rgba(245, 197, 24, 0.4);
        box-shadow: 
            0 20px 40px rgba(0, 0, 0, 0.6),
            0 0 30px rgba(245, 197, 24, 0.15),
            inset 0 0 20px rgba(255, 255, 255, 0.03);
        transform: rotateX(8deg) rotateY(-6deg) translateZ(0);
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }}

    .card::before {{
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: linear-gradient(135deg, 
            rgba(245, 197, 24, 0.15) 0%, 
            rgba(255, 215, 0, 0.05) 50%, 
            transparent 100%);
        border-radius: 24px;
        z-index: -1;
        opacity: 0;
        transition: opacity 0.4s ease;
    }}

    .card:hover {{
        transform: rotateX(4deg) rotateY(-3deg) translateY(-10px) scale(1.02);
        box-shadow: 
            0 30px 60px rgba(0, 0, 0, 0.7),
            0 0 40px rgba(245, 197, 24, 0.3);
    }}

    .card:hover::before {{
        opacity: 1;
    }}

    /* Title */
    .title {{
        font-size: 3.2rem;
        font-weight: 900;
        background: linear-gradient(90deg, #F5C518, #FFD700);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin: 0 0 0.8rem 0;
        letter-spacing: 1px;
        text-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
    }}

    .subtitle {{
        font-size: 1.35rem;
        color: #E0E0E0;
        text-align: center;
        margin-bottom: 2.5rem;
        line-height: 1.6;
        font-weight: 400;
    }}

    /* Text Area */
    .stTextArea > div > div {{
        background: rgba(0, 0, 0, 0.7) !important;
        border: 2px solid #F5C518 !important;
        border-radius: 16px !important;
    }}

    .stTextArea textarea {{
        color: #FFFFFF !important;
        font-size: 1.1rem !important;
        text-align: center !important;
        caret-color: #FFD700;
    }}

    .stTextArea textarea::placeholder {{
        color: #AAAAAA !important;
        font-style: italic;
    }}

    .stTextArea label {{
        color: #F5C518 !important;
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        text-align: center !important;
        display: block !important;
        margin-bottom: 1rem !important;
    }}

    /* Button Container */
    .button-container {{
        display: flex;
        justify-content: center;
        gap: 1.5rem;
        margin: 2rem 0;
        flex-wrap: wrap;
    }}

    /* Analyze Button */
    .analyze-btn button {{
        background: linear-gradient(90deg, #F5C518, #FFD700);
        color: #000000 !important;
        font-weight: 800;
        font-size: 1.2rem;
        padding: 0.9rem 2.8rem;
        border: none;
        border-radius: 50px;
        box-shadow: 0 8px 20px rgba(245, 197, 24, 0.4);
        transition: all 0.3s ease;
        width: 280px;
    }}

    .analyze-btn button:hover {{
        transform: translateY(-4px);
        box-shadow: 0 12px 25px rgba(245, 197, 24, 0.6);
        background: linear-gradient(45deg, #FFD700, #F5C518);
    }}

    /* Clear Button */
    .clear-btn button {{
        background: linear-gradient(90deg, #6C757D, #495057);
        color: #FFFFFF !important;
        font-weight: 700;
        font-size: 1.2rem;
        padding: 0.9rem 2.8rem;
        border: 2px solid #6C757D;
        border-radius: 50px;
        box-shadow: 0 8px 20px rgba(108, 117, 125, 0.3);
        transition: all 0.3s ease;
        width: 280px;
    }}

    .clear-btn button:hover {{
        transform: translateY(-4px);
        box-shadow: 0 12px 25px rgba(108, 117, 125, 0.5);
        background: linear-gradient(45deg, #495057, #6C757D);
        border-color: #495057;
    }}

    /* Result Card */
    .result-card {{
        background: rgba(30, 30, 30, 0.95);
        border: 2px solid;
        border-radius: 18px;
        padding: 1.8rem;
        margin: 1.8rem auto;
        text-align: center;
        max-width: 480px;
        backdrop-filter: blur(8px);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5);
    }}

    .sentiment-text {{
        font-size: 2rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }}

    .confidence-bar {{
        height: 12px;
        background: rgba(255, 255, 255, 0.2);
        border-radius: 6px;
        overflow: hidden;
        margin: 1rem 0;
        position: relative;
    }}

    .confidence-fill {{
        height: 100%;
        background: linear-gradient(90deg, #F5C518, #FFD700);
        border-radius: 6px;
        transition: width 1.2s ease;
        position: relative;
    }}

    .confidence-fill::after {{
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
        animation: shimmer 2s infinite;
    }}

    @keyframes shimmer {{
        0% {{ transform: translateX(-100%); }}
        100% {{ transform: translateX(100%); }}
    }}

    .confidence-label {{
        font-size: 1.1rem;
        color: #CCCCCC;
        margin-top: 0.8rem;
        font-weight: 500;
    }}

    /* Footer */
    .footer {{
        text-align: center;
        color: #999999;
        font-size: 0.95rem;
        margin-top: 2.5rem;
        padding-top: 1.5rem;
        border-top: 1px dashed rgba(245, 197, 24, 0.3);
    }}

    /* Hide Streamlit padding */
    .block-container {{
        padding-top: 0 !important;
    }}

    .css-1d391kg {{
        padding: 0 !important;
    }}
</style>
"""

if not image_base64:
    st.warning("⚠️ Background image not found. Using dark fallback.")
    background_style = background_style.replace(
        'background-image: url("data:image/jpg;base64,{image_base64}");',
        'background-color: #0E1117;'
    )

st.markdown(background_style, unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    model_name = "lvwerra/distilbert-imdb"
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# --- 3D CARD LAYOUT ---
st.markdown('<div class="card-container"><div class="card">', unsafe_allow_html=True)

# Header
st.markdown('<h1 class="title">🎬 Sentiment Analyzer</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Uncover the emotion behind any movie review using <b>DistilBERT</b> powered AI.</p>', unsafe_allow_html=True)

# Input
user_input = st.text_area(
    "📝 Enter your movie review:",
    placeholder="e.g., 'The acting was phenomenal and the plot kept me hooked till the end!'",
    height=140,
    label_visibility="collapsed"
)

# Button Container
st.markdown('<div class="button-container">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    analyze_btn = st.button("🔍 Analyze Sentiment", use_container_width=True, key="analyze")
    st.markdown('<div class="analyze-btn"></div>', unsafe_allow_html=True)

with col2:
    clear_btn = st.button("🗑️ Clear All", use_container_width=True, key="clear")
    st.markdown('<div class="clear-btn"></div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Session State for managing results
if 'show_result' not in st.session_state:
    st.session_state.show_result = False
if 'result_data' not in st.session_state:
    st.session_state.result_data = None

# Clear Button Logic
if clear_btn:
    st.session_state.show_result = False
    st.session_state.result_data = None
    # Clear the text area by using session state
    st.rerun()

# Analyze Button Logic
if analyze_btn and user_input.strip():
    with st.spinner("🎭 Analyzing sentiment..."):
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=256)
        outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=1).detach().cpu()
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

        sentiment = "🌟 Positive" if pred == 1 else "😠 Negative"
        color = "#00D4AA" if pred == 1 else "#FF6B6B"
        icon = "🎉" if pred == 1 else "😞"
        
        st.session_state.result_data = {
            'sentiment': sentiment,
            'color': color,
            'icon': icon,
            'confidence': confidence
        }
        st.session_state.show_result = True

# Show warning if analyze clicked with empty input
if analyze_btn and not user_input.strip():
    st.markdown(
        '<div class="result-card" style="border-color:#FF6B6B;">'
        '<p style="color:#FF6B6B; font-weight:600;">⚠️ Please enter a review to analyze.</p>'
        '</div>',
        unsafe_allow_html=True
    )

# Display Result
if st.session_state.show_result and st.session_state.result_data:
    data = st.session_state.result_data
    st.markdown(f'''
    <div class="result-card" style="border-color:{data['color']};">
        <div class="sentiment-text" style="color:{data['color']};">
            {data['icon']} {data['sentiment']}
        </div>
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: {data['confidence']*100}%;"></div>
        </div>
        <div class="confidence-label">
            Confidence: <strong>{data['confidence']:.2%}</strong>
        </div>
    </div>
    ''', unsafe_allow_html=True)

# Footer
st.markdown(
    '<p class="footer">Built with ❤️ using <b>Streamlit</b> + <b>DistilBERT (IMDb)</b></p>',
    unsafe_allow_html=True
)

st.markdown('</div></div>', unsafe_allow_html=True)

# --- HIDE STREAMLIT HEADER + FOOTER + TOP SPACE ---
st.markdown("""
    <style>
    /* Hide Streamlit's default header and hamburger menu */
    header {visibility: hidden;}
    [data-testid="stHeader"] {display: none;}
    [data-testid="stToolbar"] {display: none;}

    /* Remove top padding / spacing */
    .block-container {
        padding-top: 1rem;
    }

    /* Optionally hide the blank black strip */
    .stApp {
        background-color: transparent !important;
    }
    </style>
""", unsafe_allow_html=True)