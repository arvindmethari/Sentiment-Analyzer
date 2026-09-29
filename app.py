import os
import streamlit as st
from model_pipeline import ASTEPipeline

# Page configuration
st.set_page_config(
    page_title="Sentiment Analyzer AI",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Load custom CSS
css_file_path = os.path.join(os.path.dirname(__file__), "styles.css")
if os.path.exists(css_file_path):
    with open(css_file_path, "r", encoding="utf-8") as f:
        custom_css = f.read()
    st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)

# Cache model pipeline loading across sessions silently
@st.cache_resource(show_spinner=False)
def get_pipeline():
    return ASTEPipeline()

# Initialize session state variables
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "error_message" not in st.session_state:
    st.session_state.error_message = None

# Header Section (Strictly Centered Title & Subtitle)
st.markdown(
    """
    <div class="hero-header">
        <h1 class="hero-title">Sentiment Analyzer AI</h1>
        <p class="hero-subtitle">We decode the emotion behind every word to help you truly understand your user</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Text Area Input Section (Starts completely empty with placeholder 'Enter opinion here')
user_review = st.text_area(
    label="Product Review Input",
    placeholder="Enter opinion here",
    height=150,
    label_visibility="collapsed",
    key="review_text"
)

# Dynamic Placeholder for Error Message (renders right below text box)
error_placeholder = st.empty()

# Centered Pill-shaped CTA Button
btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])
with btn_col2:
    analyze_clicked = st.button("Analyze", use_container_width=True)

# Handle Analyze action
if analyze_clicked:
    cleaned_input = user_review.strip()
    if not cleaned_input or cleaned_input.lower() in ["enter opinion here", "enter your opinion here"]:
        st.session_state.error_message = "Please enter the opinion"
        st.session_state.analysis_result = None
    else:
        st.session_state.error_message = None
        non_ascii = len([c for c in cleaned_input if ord(c) > 127])
        is_multilingual = (non_ascii / max(len(cleaned_input), 1)) > 0.15
        spinner_text = "Translating & Processing..." if is_multilingual else "Processing..."
        with st.spinner(spinner_text):
            pipeline = get_pipeline()
            st.session_state.analysis_result = pipeline.analyze(cleaned_input)

# Display error message in red if present
if st.session_state.error_message:
    error_placeholder.markdown(
        f'<div class="error-msg">⚠️ {st.session_state.error_message}</div>',
        unsafe_allow_html=True
    )

# Output Display Box
res = st.session_state.analysis_result

if res is None:
    # Initial State Box
    st.markdown(
        """
        <div class="output-container-initial">
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#4B5563" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" style="margin: 0 auto 12px auto; display: block;">
                <circle cx="12" cy="12" r="10"></circle>
                <line x1="12" y1="16" x2="12" y2="12"></line>
                <line x1="12" y1="8" x2="12.01" y2="8"></line>
            </svg>
            Analysis results will appear here after clicking Analyze.
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    # Active Glow Output Container
    st.markdown('<div class="output-container-active">', unsafe_allow_html=True)
    
    # Multilingual Translation Banner (if applicable)
    if res.get("was_translated"):
        src_lang = res.get("detected_lang", "Unknown").upper()
        st.markdown(
            f"""
            <div class="translation-notice">
                <span style="font-size: 1.1rem;">🌐</span>
                <div>
                    <strong>Multilingual Input Detected ({src_lang}):</strong> Translated to English:<br/>
                    <em style="color: #A5F3FC;">"{res.get('english_text')}"</em>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    triplets = res.get("results", [])
    
    # Stats Overview Chips
    pos_count = sum(1 for t in triplets if t["sentiment"] == "Positive")
    neg_count = sum(1 for t in triplets if t["sentiment"] == "Negative")
    neu_count = sum(1 for t in triplets if t["sentiment"] == "Neutral")
    
    st.markdown(
        f"""
        <div class="stats-bar">
            <div class="stat-chip">🎯 Extracted Aspects: <b>{len(triplets)}</b></div>
            <div class="stat-chip">🟢 Positive: <b style="color: #4ADE80;">{pos_count}</b></div>
            <div class="stat-chip">🔴 Negative: <b style="color: #F87171;">{neg_count}</b></div>
            <div class="stat-chip">🔵 Neutral: <b style="color: #38BDF8;">{neu_count}</b></div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Render Structured Aspect Cards
    for item in triplets:
        aspect = item["aspect"]
        sentiment = item["sentiment"]
        opinion = item["opinion"]
        explanation = item["explanation"]

        if sentiment == "Positive":
            badge_class = "badge-positive"
            badge_icon = "●"
        elif sentiment == "Negative":
            badge_class = "badge-negative"
            badge_icon = "▲"
        else:
            badge_class = "badge-neutral"
            badge_icon = "■"

        st.markdown(
            f"""
            <div class="aspect-card">
                <div class="aspect-header">
                    <div class="aspect-title">
                        <span style="color: #22C55E;">✦</span> {aspect}
                    </div>
                    <div style="display: flex; gap: 8px; align-items: center;">
                        <span class="opinion-badge">Opinion: "{opinion}"</span>
                        <span class="{badge_class}">{badge_icon} {sentiment}</span>
                    </div>
                </div>
                <p class="explanation-text">{explanation}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)
