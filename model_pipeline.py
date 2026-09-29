import os
import re
import difflib
import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple
from transformers import AutoConfig, DebertaV2Model, AutoTokenizer

ASTE_MODEL_DIR = r"G:\JYESTA\Projects\Minor\aste_model"
ASTE_WEIGHTS_PATH = os.path.join(ASTE_MODEL_DIR, "deberta_aste_weights.pt")
TRANSLATION_MODEL_DIR = r"G:\JYESTA\Projects\Minor\Translation\nllb_200_model"

ID2LABEL = {
    0: "O",
    1: "B-Aspect",
    2: "I-Aspect",
    3: "B-Opinion",
    4: "I-Opinion"
}

LANG_TO_NLLB = {
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "hi": "hin_Deva",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "ru": "rus_Cyrl",
    "zh-cn": "zho_Hans",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "ar": "arb_Arab",
    "te": "tel_Telu",
    "ta": "tam_Taml",
    "bn": "ben_Beng",
    "mr": "mar_Deva",
    "gu": "guj_Gujr",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "pa": "pan_Guru",
    "ur": "urd_Arab",
    "nl": "nld_Latn",
    "tr": "tur_Latn",
    "vi": "vie_Latn",
    "id": "ind_Latn",
}

POSITIVE_LEXICON = {
    "great", "amazing", "excellent", "superb", "awesome", "fantastic", "good", "wonderful",
    "brilliant", "outstanding", "exceptional", "breathtaking", "crisp", "vibrant", "sharp",
    "fast", "quick", "responsive", "smooth", "clear", "loud", "punchy", "rich", "solid",
    "durable", "sturdy", "premium", "sleek", "beautiful", "gorgeous", "comfortable", "perfect",
    "flawless", "reliable", "efficient", "intuitive", "user-friendly", "convenient", "impressive",
    "top-notch", "love", "loved", "liked", "enjoyed", "best", "super", "decent", "fine",
    "stellar", "terrific", "phenomenal", "pleased", "satisfied", "worth", "stunning", "bright"
}

NEGATIVE_LEXICON = {
    "bad", "terrible", "horrible", "awful", "poor", "weak", "slow", "sluggish", "laggy",
    "lag", "freezes", "crash", "crashes", "crap", "garbage", "trash", "dull", "blurry",
    "washed", "faint", "quiet", "muffled", "distorted", "cheap", "fragile", "flimsy",
    "broken", "broke", "defective", "ugly", "uncomfortable", "painful", "heavy", "bulky",
    "drain", "drains", "draining", "short", "overheats", "hot", "expensive", "overpriced",
    "waste", "useless", "disappointed", "disappointing", "worst", "buggy", "hated", "annoying",
    "pesima", "pesimo", "malo", "mala", "horrible", "defectuoso", "frustrating", "mediocre"
}

NEGATIONS = {
    "not", "never", "no", "hardly", "barely", "scarcely", "without", "rarely",
    "doesn't", "don't", "didn't", "isn't", "aren't", "wasn't", "weren't", "won't", "cannot", "can't"
}

INTENSIFIERS = {"very", "extremely", "really", "incredibly", "super", "exceptionally", "absolutely", "totally", "quite"}

CANONICAL_ASPECTS = [
    "display picture quality", "picture quality", "battery life", "battery",
    "sound quality", "sound", "audio", "speaker", "speakers", "screen", "display",
    "camera", "performance", "speed", "build quality", "build", "design", "keyboard",
    "trackpad", "price", "value", "customer service", "delivery", "packaging",
    "size", "weight", "software", "app", "comfort", "materials",
    "phone", "smartphone", "mobile", "device", "hardware", "laptop", "tablet",
    "headphones", "earphones", "earbuds", "charger", "charging", "battery backup"
]

class DebertaASTE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.deberta = DebertaV2Model(config)
        self.bio_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 384),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(384, 5)
        )

    def forward(self, input_ids, attention_mask=None, **kwargs):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        logits = self.bio_classifier(sequence_output)
        return logits


class ASTEPipeline:
    def __init__(self):
        self.aste_model = None
        self.aste_tokenizer = None
        self.trans_model = None
        self.trans_tokenizer = None
        self._load_aste_model()

    def _load_aste_model(self):
        """Loads ASTE DeBERTa model into memory."""
        config = AutoConfig.from_pretrained(ASTE_MODEL_DIR)
        self.aste_tokenizer = AutoTokenizer.from_pretrained(ASTE_MODEL_DIR)
        self.aste_model = DebertaASTE(config)
        state_dict = torch.load(ASTE_WEIGHTS_PATH, map_location="cpu")
        self.aste_model.load_state_dict(state_dict, strict=True)
        self.aste_model.eval()

    def _get_translation_model(self):
        """Lazy load NLLB-200 only when non-English text is received."""
        if self.trans_model is None:
            from transformers import AutoModelForSeq2SeqLM
            self.trans_tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL_DIR)
            self.trans_model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL_DIR)
            self.trans_model.eval()
        return self.trans_tokenizer, self.trans_model

    def detect_and_translate(self, text: str) -> Tuple[str, bool, str]:
        text = text.strip()
        if not text:
            return text, False, "en"

        detected_lang = "en"
        try:
            from langdetect import detect
            detected_lang = detect(text)
        except Exception:
            non_ascii = len([c for c in text if ord(c) > 127])
            if non_ascii / max(len(text), 1) > 0.15:
                detected_lang = "es"

        if detected_lang == "en":
            return text, False, "en"

        src_lang_code = LANG_TO_NLLB.get(detected_lang, "spa_Latn")
        try:
            tokenizer, model = self._get_translation_model()
            tokenizer.src_lang = src_lang_code
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            forced_bos_token_id = tokenizer.convert_tokens_to_ids("eng_Latn")
            
            with torch.inference_mode():
                generated_tokens = model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_new_tokens=48,
                    num_beams=1,
                    do_sample=False
                )
            translated = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            return translated, True, detected_lang
        except Exception:
            return text, False, detected_lang

    def _extract_spans_from_bio(self, tokens: List[str], preds: List[int]) -> Tuple[List[str], List[str]]:
        aspects = []
        opinions = []
        
        curr_aspect = []
        curr_opinion = []

        for token, pred in zip(tokens, preds):
            label = ID2LABEL.get(pred, "O")
            clean_token = token.replace("\u2581", "").strip()
            if not clean_token or clean_token in {"[CLS]", "[SEP]", "[PAD]"}:
                continue

            if label in {"B-Aspect", "I-Aspect"}:
                curr_aspect.append(clean_token)
            else:
                if curr_aspect:
                    asp_phrase = " ".join(curr_aspect)
                    if len(asp_phrase) > 1:
                        aspects.append(asp_phrase)
                    curr_aspect = []

            if label in {"B-Opinion", "I-Opinion"}:
                curr_opinion.append(clean_token)
            else:
                if curr_opinion:
                    op_phrase = " ".join(curr_opinion)
                    if len(op_phrase) > 1:
                        opinions.append(op_phrase)
                    curr_opinion = []

        if curr_aspect:
            asp_phrase = " ".join(curr_aspect)
            if len(asp_phrase) > 1:
                aspects.append(asp_phrase)
        if curr_opinion:
            op_phrase = " ".join(curr_opinion)
            if len(op_phrase) > 1:
                opinions.append(op_phrase)

        return aspects, opinions

    def _fuzzy_and_grammatical_aspects(self, text: str) -> List[str]:
        """
        Extracts aspects using:
        1. Exact keyword match
        2. Fuzzy typo-tolerant match (e.g. 'batter' -> 'Battery', 'dispaly' -> 'Display')
        3. Clause-level grammatical subject extraction (e.g. '[the batter] is not good')
        """
        found = []
        lower_text = text.lower()
        words = re.findall(r"\b[a-zA-Z']+\b", lower_text)

        # 1. Exact matches for canonical aspects
        for aspect in CANONICAL_ASPECTS:
            if re.search(r"\b" + re.escape(aspect) + r"\b", lower_text):
                found.append(aspect.title())

        # 2. Fuzzy typo matching (e.g. 'batter' -> 'battery', 'chargr' -> 'charger')
        stop_words = {"the", "a", "an", "is", "was", "are", "were", "and", "but", "not", "too", "so"}
        for word in words:
            if len(word) >= 4 and word not in stop_words and word not in POSITIVE_LEXICON and word not in NEGATIVE_LEXICON:
                # Check if it closely matches any canonical aspect word
                for canonical in CANONICAL_ASPECTS:
                    canonical_words = canonical.split()
                    matches = difflib.get_close_matches(word, canonical_words, n=1, cutoff=0.72)
                    if matches:
                        # Found typo match! Replace or append the canonical title
                        found.append(matches[0].title())

        # 3. Syntactic Clause-Subject Discovery:
        # Matches patterns like 'the <noun phrase> is/was/are/has/feels/looks <opinion>'
        clauses = re.split(r"(?:,|\.|\bbut\b|\bhowever\b|\balthough\b|\bwhile\b|\byet\b|\band\b|;)+", text, flags=re.IGNORECASE)
        for clause in clauses:
            clause = clause.strip()
            # Pattern: (the/my/this)? [Subject] (is|are|was|were|seems|looks|feels|has)
            m = re.search(r"(?:^|\s)(?:the\s+|my\s+|this\s+)?([a-zA-Z\s]{2,25}?)\s+(?:is|are|was|were|has|seems|looks|feels)\b", clause, flags=re.IGNORECASE)
            if m:
                subj = m.group(1).strip()
                # Ensure subj is not just a pronoun/filler
                if len(subj) >= 3 and subj.lower() not in {"it", "that", "this", "there", "everything", "nothing", "something"}:
                    # Check if fuzzy matches a known aspect
                    matches = difflib.get_close_matches(subj.lower(), CANONICAL_ASPECTS, n=1, cutoff=0.70)
                    if matches:
                        found.append(matches[0].title())
                    else:
                        found.append(subj.title())

        return found

    def _consolidate_aspects(self, text: str, candidate_aspects: List[str]) -> List[str]:
        """
        Merges adjacent or overlapping aspect fragments into maximal compound aspect phrases.
        For instance: 'Phone' + 'Display' -> 'Phone Display'
        'Display' + 'Picture Quality' -> 'Display Picture Quality'
        'Battery' + 'Life' -> 'Battery Life'
        """
        lower_text = text.lower()
        spans = []
        for asp in candidate_aspects:
            asp_clean = asp.strip().lower()
            if not asp_clean or len(asp_clean) < 2:
                continue
            
            # Check exact occurrence in sentence
            matched_any = False
            for m in re.finditer(r"\b" + re.escape(asp_clean) + r"\b", lower_text):
                spans.append((m.start(), m.end(), asp.title()))
                matched_any = True
                
            # If not exact (e.g. fuzzy corrected like 'battery' for 'batter'), find the source typo in text
            if not matched_any:
                words = re.findall(r"\b[a-zA-Z']+\b", lower_text)
                for w in words:
                    if difflib.get_close_matches(w, [asp_clean], n=1, cutoff=0.72):
                        for m in re.finditer(r"\b" + re.escape(w) + r"\b", lower_text):
                            spans.append((m.start(), m.end(), asp.title()))

        if not spans:
            return []

        # Sort by start position ascending, then length descending
        spans = sorted(spans, key=lambda x: (x[0], -(x[1] - x[0])))

        merged = []
        for cur_start, cur_end, label in spans:
            if not merged:
                merged.append([cur_start, cur_end, label])
                continue
            prev_start, prev_end, prev_label = merged[-1]

            # Check if this span is already inside the previous span
            if cur_start >= prev_start and cur_end <= prev_end:
                continue

            # Check if adjacent or overlapping (separated only by spaces or hyphens)
            gap = lower_text[prev_end:cur_start]
            if cur_start <= prev_end or (cur_start > prev_end and re.match(r"^[ -]+$", gap)):
                merged[-1][1] = max(prev_end, cur_end)
                merged[-1][2] = f"{prev_label} {label}"
            else:
                merged.append([cur_start, cur_end, label])

        results = []
        seen = set()
        for s, e, fallback_label in merged:
            extracted = text[s:e].strip()
            # If extracted phrase is a typo, map to clean title
            cleaned = extracted.title()
            for canonical in CANONICAL_ASPECTS:
                if difflib.get_close_matches(extracted.lower(), [canonical], n=1, cutoff=0.72):
                    cleaned = canonical.title()
                    break

            if cleaned.lower() not in seen and len(cleaned) > 1:
                seen.add(cleaned.lower())
                results.append(cleaned)
        return results

    def _extract_aspect_clause(self, text: str, aspect: str) -> str:
        clauses = re.split(r"(?:,|\.|\bbut\b|\bhowever\b|\balthough\b|\bwhile\b|\byet\b|\band\b|;)+", text, flags=re.IGNORECASE)
        asp_lower = aspect.lower()
        
        # Check direct or fuzzy occurrence in clause
        for clause in clauses:
            if asp_lower in clause.lower():
                return clause.strip()
            words = re.findall(r"\b[a-zA-Z']+\b", clause.lower())
            for w in words:
                if difflib.get_close_matches(w, [asp_lower], n=1, cutoff=0.70):
                    return clause.strip()
                for aw in asp_lower.split():
                    if difflib.get_close_matches(w, [aw], n=1, cutoff=0.70):
                        return clause.strip()
        return text

    def _calculate_sentiment(self, clause_text: str, aspect: str, opinion_phrase: str = "") -> Tuple[str, float, str, str]:
        if opinion_phrase and opinion_phrase.lower() not in clause_text.lower():
            eval_text = (opinion_phrase + " " + clause_text).lower()
        else:
            eval_text = clause_text.lower()
        words = re.findall(r"\b[a-zA-Z']+\b", eval_text)
        
        score = 0.0
        pos_hits = []
        neg_hits = []
        is_negated = False

        for i, word in enumerate(words):
            if word in NEGATIONS:
                is_negated = True
                continue

            weight = 1.0
            if i > 0 and words[i-1] in INTENSIFIERS:
                weight = 1.5

            if word in POSITIVE_LEXICON:
                delta = weight * (-1.0 if is_negated else 1.0)
                score += delta
                if is_negated:
                    neg_hits.append(f"not {word}")
                else:
                    pos_hits.append(word)
                is_negated = False
            elif word in NEGATIVE_LEXICON:
                delta = weight * (1.0 if is_negated else -1.0)
                score += delta
                if is_negated:
                    pos_hits.append(f"not {word}")
                else:
                    neg_hits.append(word)
                is_negated = False

        active_opinion = opinion_phrase
        if not active_opinion:
            if neg_hits:
                active_opinion = ", ".join(neg_hits[:2])
            elif pos_hits:
                active_opinion = ", ".join(pos_hits[:2])
            else:
                active_opinion = "contextual feedback"

        if score > 0.1:
            sentiment = "Positive"
            cues = ", ".join(f"'{w}'" for w in pos_hits[:2]) or f"'{active_opinion}'"
            explanation = (
                f"The aspect **{aspect}** was assigned a **Positive** sentiment because the user expressed "
                f"favorable feedback highlighting {cues}, indicating strong satisfaction with this attribute."
            )
        elif score < -0.1:
            sentiment = "Negative"
            cues = ", ".join(f"'{w}'" for w in neg_hits[:2]) or f"'{active_opinion}'"
            explanation = (
                f"The aspect **{aspect}** was assigned a **Negative** sentiment because the user expressed "
                f"dissatisfaction characterized by {cues}, pointing to critical flaws or underwhelming performance."
            )
        else:
            sentiment = "Neutral"
            explanation = (
                f"The aspect **{aspect}** was assigned a **Neutral** sentiment as the review presents a balanced, "
                f"descriptive, or impartial observation ({active_opinion}) without marked positive or negative polarity."
            )

        return sentiment, score, active_opinion, explanation

    def analyze(self, raw_text: str) -> Dict[str, Any]:
        raw_text = raw_text.strip()
        if not raw_text:
            return {
                "original_text": "",
                "english_text": "",
                "was_translated": False,
                "detected_lang": "en",
                "results": []
            }

        english_text, was_translated, detected_lang = self.detect_and_translate(raw_text)

        # 1. ASTE Model Sequence Tagging
        inputs = self.aste_tokenizer(english_text, return_tensors="pt")
        with torch.no_grad():
            logits = self.aste_model(**inputs)
            preds = torch.argmax(logits, dim=-1)[0].tolist()

        tokens = self.aste_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        raw_aspects, raw_opinions = self._extract_spans_from_bio(tokens, preds)

        # 2. Add domain, fuzzy typo-resilient, and clause-syntactic aspects
        extended_candidates = list(raw_aspects) + self._fuzzy_and_grammatical_aspects(english_text)

        # 3. Unify adjacent and overlapping words into maximal compound aspect phrases
        consolidated_aspects = self._consolidate_aspects(english_text, extended_candidates)

        triplets = []
        for asp in consolidated_aspects:
            clause = self._extract_aspect_clause(english_text, asp)
            
            # Match opinion if in clause
            matched_op = ""
            for op in raw_opinions:
                if op.lower() in clause.lower():
                    matched_op = op
                    break
            
            sentiment, score, active_op, explanation = self._calculate_sentiment(clause, asp, matched_op)
            triplets.append({
                "aspect": asp,
                "opinion": active_op,
                "sentiment": sentiment,
                "score": round(score, 2),
                "explanation": explanation
            })

        # Fallback to general sentiment if no specific aspect detected
        if not triplets:
            sentiment, score, active_op, explanation = self._calculate_sentiment(english_text, "Overall Experience")
            triplets.append({
                "aspect": "Overall Experience",
                "opinion": active_op,
                "sentiment": sentiment,
                "score": round(score, 2),
                "explanation": explanation
            })

        return {
            "original_text": raw_text,
            "english_text": english_text,
            "was_translated": was_translated,
            "detected_lang": detected_lang,
            "results": triplets
        }
