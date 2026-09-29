# Sentiment Analyzer AI (ABSA Web Application)

A modern, ultra-sleek, dark-mode Aspect-Based Sentiment Analysis (ABSA) web application inspired by the developer-first aesthetic of [Modal.com](https://modal.com).

Featuring a solid dark background (`#06090E`), crisp typography, glowing neon green/cyan accents, and smooth micro-interactions.

---

## Key Features

1. **Modal.com Aesthetic & Micro-Interactions**:
   - Deep void background (`#06090E`) with subtle dot-matrix radial gradients.
   - Dynamic border glow on input text area with smooth CSS transitions (`transition: all 0.3s ease-in-out`).
   - Neon green (`#22C55E`) pill-shaped "Analyze" CTA button with pulsing glow shadow and scale micro-interaction.
   - Glowing animated output container and structured aspect cards.
   - Color-coded sentiment badges (Positive: Neon Green `#22C55E`, Negative: Neon Red `#EF4444`, Neutral: Electric Cyan `#38BDF8`).

2. **Dual-Engine Local AI Pipeline**:
   - **ASTE (Aspect Sentiment Triplet Extraction)**: Fine-tuned Dual-Head DeBERTa-v3 model (`G:\JYESTA\Projects\Minor\aste_model\deberta_aste_weights.pt`) loaded 100% locally from disk.
   - **NLLB-200 Multilingual Translation**: Local NLLB-200 model (`G:\JYESTA\Projects\Minor\Translation\nllb_200_model`) automatically translates non-English reviews (Spanish, French, German, Hindi, etc.) into English before ASTE processing.
   - **Contextual Natural Language Explanations**: Every extracted aspect is accompanied by an in-depth rationale explaining why the sentiment was assigned based on the reviewer's opinion cues.

3. **Zero Internet Dependence**:
   - Models and tokenizers are loaded strictly from local disk.
   - The model archives are intentionally not committed because they exceed GitHub's 100 MB per-file limit. Place the extracted model directories at the paths configured in `model_pipeline.py` before running the app.

---

## Directory Structure

```
G:\JYESTA\Projects\Minor\Sentiment_Analyzer\
├── app.py                 # Streamlit Web Application
├── model_pipeline.py      # DeBERTa ASTE + NLLB Translation Backend Pipeline
├── styles.css             # Modal.com-inspired CSS styling & animations
├── requirements.txt       # Project dependencies
├── run_app.bat            # Double-click launcher script
└── .venv/                 # Python 3.12 Virtual Environment
```

---

## How to Run

### Option 1: Double-Click Launcher
Simply double-click `run_app.bat`.

### Option 2: Command Line
```powershell
cd G:\JYESTA\Projects\Minor\Sentiment_Analyzer
.venv\Scripts\activate
streamlit run app.py
```
Open your browser at `http://localhost:8501`.
