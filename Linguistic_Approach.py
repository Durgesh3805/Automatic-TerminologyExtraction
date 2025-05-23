import re
import time
from collections import Counter
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# ---------- Load Transformer Model for POS Tagging ----------
model_name = "vblagoje/bert-english-uncased-finetuned-pos"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
pos_tagger = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# ---------- POS Patterns to Match ----------
POS_PATTERNS = [
    ['ADJ', 'NOUN'],
    ['NOUN', 'NOUN'],
    ['ADJ', 'NOUN', 'NOUN'],
    ['NOUN', 'ADP', 'NOUN']
]

# ---------- Preprocessing ----------
def preprocess(text):
    """
    Clean and preprocess the text by converting it to lowercase and removing non-alphabetical characters.
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()

# ---------- POS Tagging ----------
def get_lemmas_and_pos(text, pos_tagger):
    """
    Tokenizes and tags parts of speech (POS) using the given POS tagger model.
    """
    tagged = pos_tagger(text)
    words = [t['word'].lower() for t in tagged]
    pos_tags = [t['entity_group'] for t in tagged]
    return words, pos_tags

# ---------- Match POS Patterns ----------
def match_patterns(words, pos_tags, patterns):
    """
    Matches POS patterns in the tokenized text.
    """
    matched = []
    for i in range(len(words)):
        for pattern in patterns:
            window = len(pattern)
            if i + window <= len(pos_tags) and pos_tags[i:i + window] == pattern:
                phrase = " ".join(words[i:i + window])
                matched.append((phrase, pattern))
    return matched

# ---------- Unithood Score ----------
def unithood_score(phrase, doc_freqs):
    """
    Calculate the unithood score for a given phrase.
    """
    score = 0
    freq = doc_freqs[phrase]

    # 1 point for multiple-word phrases
    if len(phrase.split()) > 1:
        score += 1

    # 1 point if frequent
    if freq >= 2:
        score += 1

    # 1 point if contains a noun-like structure
    if re.search(r"\b[a-z]+\s[a-z]+", phrase):
        score += 1

    return score

# ---------- Termhood Score ----------
def termhood_score(phrase):
    """
    Calculate the termhood score for a given phrase.
    """
    score = 0
    stop_words = {"the", "and", "in", "of", "no", "on", "for", "a", "with", "to"}
    tokens = phrase.split()

    # 1 point if phrase has 2+ words
    if len(tokens) > 1:
        score += 1

    # 1 point if none of the tokens are stopwords
    if not any(w in stop_words for w in tokens):
        score += 1

    # 1 point if phrase has medical-like structure
    if any(re.search(r"(itis|oma|algia|emia|osis|pathy|plasty|ectomy|uria)", t) for t in tokens):
        score += 1

    return score

# ---------- Main Term Extraction Function ----------
def extract_terms_verbose(emr_texts, threshold_u=0.5, threshold_t=0.5):
    """
    Extract terms from a list of EMR (Electronic Medical Record) texts, match POS patterns, 
    calculate unithood and termhood scores, and return the final extracted terms.
    """
    all_phrases = []
    all_matches = []

    print("\n🔹 Preprocessing Output\n")
    for text in emr_texts:
        print(f"Original: {text}")
        clean_text = preprocess(text)
        words, pos_tags = get_lemmas_and_pos(clean_text, pos_tagger)
        phrases = match_patterns(words, pos_tags, POS_PATTERNS)
        simplified = sorted(set([p[0] for p in phrases]))
        all_phrases.extend(simplified)
        all_matches.extend(phrases)
        for s in simplified:
            print(f"- {s}")
        print()

    doc_freqs = Counter(all_phrases)

    print("\n🔹 Matched Patterns\n")
    for phrase, tags in all_matches:
        print(f"{phrase:<30} {tags}")

    print("\n🔹 Scoring (Unithood & Termhood)\n")
    results = []
    for phrase in sorted(set(all_phrases)):
        u_score = unithood_score(phrase, doc_freqs)
        t_score = termhood_score(phrase)
        accepted = u_score >= threshold_u and t_score >= threshold_t
        results.append((phrase, u_score, t_score, accepted))
        print(f"{phrase:<30} U: {u_score} | T: {t_score} | {'✅' if accepted else '❌'}")

    final_terms = [r[0] for r in results if r[3]]

    print("\n✅ Final Extracted Terms")
    print(final_terms)
    return final_terms

# ---------- Evaluation Function ----------
def evaluate_extraction(emr_texts, ground_truth):
    """
    Evaluates the term extraction process by comparing predicted terms with ground truth data.
    """
    start_time = time.time()
    predicted_terms = extract_terms_verbose(emr_texts)
    end_time = time.time()

    pred_set = set(predicted_terms)
    true_set = set([item for sublist in ground_truth for item in sublist])

    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1)
    exec_time = end_time - start_time

    print("\n🔍 Evaluation Results")
    print(f"True Positives : {tp}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"Precision      : {precision:.4f}")
    print(f"Recall         : {recall:.4f}")
    print(f"F1-Score       : {f1:.4f}")
    print(f"Execution Time : {exec_time:.2f} seconds")

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'execution_time': exec_time
    }

# ---------- Example Usage ----------
if __name__ == "__main__":
    emr_texts = [
        "Patient reports chest pain radiating to the left arm and jaw, suggestive of angina.",
        "Diagnosed with type 2 diabetes mellitus and prescribed metformin 500mg twice daily.",
        "Complains of blurred vision and frequent urination over the past few days."
    ]
    ground_truth = [
        ["chest pain", "left arm", "jaw", "angina"],
        ["type 2 diabetes mellitus", "metformin"],
        ["blurred vision", "frequent urination"]
    ]
    evaluate_extraction(emr_texts, ground_truth)
