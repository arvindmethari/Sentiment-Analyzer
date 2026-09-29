<div align="center">

# 🧠 Sentiment Analyzer

**Aspect-level sentiment analysis with multilingual support and real-time insights**

[![Python](https://img.shields.io/badge/Python-3.12+-blue?style=flat-square&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io)
[![DeBERTa](https://img.shields.io/badge/DeBERTa--v3-Microsoft-0078D4?style=flat-square&logo=microsoft)](https://huggingface.co/microsoft/deberta-v3-base)
[![NLLB-200](https://img.shields.io/badge/NLLB--200-Meta%20AI-1877F2?style=flat-square&logo=meta)](https://ai.meta.com/research/no-language-left-behind/)
[![Offline](https://img.shields.io/badge/Runs-100%25%20Offline-brightgreen?style=flat-square)](https://github.com)

*Understanding what people feel — about every aspect — in any language.*

</div>

---

## 📖 Overview

SentimentVision is a full-stack AI application that performs **Aspect-Based Sentiment Analysis (ABSA)** on product and service reviews. It:

- 🎯 **Extracts specific aspects** from a review (e.g., *Display*, *Battery Life*, *Camera*)
- 💬 **Detects sentiment** for each aspect individually — Positive, Negative, or Neutral
- 🌐 **Translates input automatically** — supports Telugu, Hindi, and 200+ languages via NLLB-200
- 🔍 **Handles typos gracefully** using fuzzy matching (`batter` → `Battery`)
- ⚡ **Runs fully offline** — no API keys, no internet, no cloud inference
- 🎨 **Serves a dark-themed Streamlit UI** inspired by Modal.com

---

## 🏗️ Architecture

```
Sentiment_Analyzer/
│
├── app.py                      # Streamlit UI — hero header, textarea, output cards
├── model_pipeline.py           # Core ML pipeline — ASTE + translation + fuzzy logic
├── styles.css                  # Dark theme CSS — glowing input, neon accents, badge styles
├── run_app.bat                 # One-click launcher (double-click to run)
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── .streamlit/
│   └── config.toml             # Native dark theme config (background, text color, accent)
│
└── Models/                     # AI model weights (see download instructions below)
    ├── aste_model.zip          # DeBERTa-v3 ASTE weights — extract to ../aste_model/
    └── Translation.zip         # NLLB-200 translation model — extract to ../Translation/
```

**Model files live outside the app folder (not committed to git):**

```
Projects/Minor/
├── aste_model/
│   ├── deberta_aste_weights.pt     # Custom-trained DeBERTa-v3 weights (736 MB)
│   ├── config.json                 # DeBERTa-v3-base model config
│   ├── tokenizer.json              # SentencePiece tokenizer
│   └── tokenizer_config.json
│
└── Translation/
    └── nllb_200_model/
        ├── model.safetensors       # NLLB-200 weights (2.46 GB)
        ├── config.json
        ├── tokenizer.json
        └── tokenizer_config.json
```

---

## ⚡ Quick Start

### Prerequisites

- Python 3.12+
- ~4 GB free RAM
- ~3.5 GB disk space (for models)

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Sentiment_Analyzer.git
cd Sentiment_Analyzer
```

### 2. Download the AI Models

> Extract `Models/aste_model.zip` → `../aste_model/`
> Extract `Models/Translation.zip` → `../Translation/nllb_200_model/`

Verify the final paths:
- `..\aste_model\deberta_aste_weights.pt`
- `..\Translation\nllb_200_model\model.safetensors`

### 3. Create a Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Run the App

**Option A — Double-click launcher:**
```
run_app.bat
```

**Option B — Terminal:**
```bash
.venv\Scripts\streamlit.exe run app.py
```

Open your browser at **[http://localhost:8501](https://sentiment-analyzer-6ifxaw5h77hxxllj56p8db.streamlit.app/)** 🚀

---

## 🧠 AI Models

### 1. ASTE Model — Aspect-Sentiment-Triple Extraction

| Property | Details |
|---|---|
| **Architecture** | DeBERTa-v3-base (Microsoft) |
| **Task** | BIO sequence labeling (Aspect + Opinion span extraction) |
| **Training Data** | Custom product review dataset |
| **Weights** | `deberta_aste_weights.pt` (736 MB) |
| **Inference** | CPU — ~1–3 seconds per review |

**BIO Label Mapping:**

| Label | Meaning |
|---|---|
| `O` | Outside (not an aspect/opinion word) |
| `B-Aspect` | Beginning of an aspect span |
| `I-Aspect` | Inside an aspect span |
| `B-Opinion` | Beginning of an opinion span |
| `I-Opinion` | Inside an opinion span |

### 2. Translation Model — NLLB-200 (Meta AI)

| Property | Details |
|---|---|
| **Architecture** | M2M100 (encoder-decoder) |
| **Languages** | 200+ including Telugu, Hindi, Tamil, Arabic, etc. |
| **Weights** | `model.safetensors` (2.46 GB) |
| **Inference** | CPU — ~30–60 seconds per review (lazy-loaded) |
| **Loading** | On-demand — only loads when non-English input is detected |

---

## 🔍 How It Works

```
User Input (any language)
        │
        ▼
  Language Detection (langdetect)
        │
   Non-English? ──── Yes ────▶ NLLB-200 Translation ──▶ English Text
        │                                                       │
       No ◀─────────────────────────────────────────────────────┘
        │
        ▼
 DeBERTa-v3 Tokenization
        │
        ▼
 BIO Sequence Labeling (Aspect & Opinion spans)
        │
        ▼
 Span Consolidation (compound aspects: "Battery" + "Life" → "Battery Life")
        │
        ▼
 Fuzzy Typo Correction ("batter" → "Battery", cutoff = 0.82)
        │
        ▼
 Sentiment Scoring (clause isolation + negation detection)
        │
        ▼
 Output: [Aspect | Sentiment | Confidence Score]
```

---

## 🎯 Supported Aspect Categories

| Category | Example Aspects |
|---|---|
| 📱 Display | Screen, Display, Picture Quality |
| 🔋 Battery | Battery, Battery Life |
| 📷 Camera | Camera, Front Camera, Rear Camera |
| ⚡ Performance | Processor, Speed, RAM, Performance |
| 🔊 Audio | Speaker, Sound, Volume |
| 🏗️ Build | Build Quality, Design, Body |
| 💰 Price | Price, Value, Cost |

---

## 🌐 Multilingual Support

SentimentVision automatically detects and translates 200+ languages before analysis:

| Language | Script | Example Input |
|---|---|---|
| Telugu | తెలుగు | ఫోన్ డిస్‌ప్లే చాలా బాగుంది కానీ బ్యాటరీ అంత బాగా లేదు |
| Hindi | हिन्दी | फ़ोन की डिस्प्ले अच्छी है पर बैटरी ख़राब है |
| Tamil | தமிழ் | திரை மிகவும் நல்லது ஆனால் பேட்டரி மோசம் |
| English | Latin | The display is great but battery life is poor |

---

## 🎨 Tech Stack

| Layer | Technology |
|---|---|
| **NLP / AI** | DeBERTa-v3-base, PyTorch 2.x |
| **Translation** | NLLB-200 (Meta AI), Transformers 5.x |
| **Language Detection** | `langdetect` |
| **Fuzzy Matching** | Python `difflib` |
| **Frontend** | Streamlit, Custom CSS (Modal.com dark theme) |
| **Tokenization** | HuggingFace Transformers, SentencePiece |
| **Environment** | Python 3.12, `uv` / `pip` |

---

## 📊 Example Output

**Input:** `"Phone display is good but the battery is not good"`

| Aspect | Sentiment | Score |
|---|---|---|
| Display | ✅ Positive | +1.00 |
| Battery | ❌ Negative | −1.00 |

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">
  Because average sentiment ratings never tell the full story. 🎯
</div>
