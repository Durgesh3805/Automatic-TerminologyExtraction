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
# POS_PATTERNS = [
#     # ['NOUN'],                # Single noun
#     ['ADJ', 'NOUN'],         # Adjective + Noun
#     ['NOUN', 'NOUN'],        # Noun + Noun
#     # ['ADJ', 'NOUN', 'NOUN'], # Adjective + Noun + Noun
#     # ['NOUN', 'ADP', 'NOUN'], # Noun + Preposition + Noun
#     # ['NOUN', 'CCONJ', 'NOUN'], # Noun + Conjunction + Noun (e.g., "spleen and pancreas")
#     # ['ADJ', 'ADJ', 'NOUN'],  # Adjective + Adjective + Noun (e.g., "big white brain")
#     # ['NOUN', 'ADJ']          # Noun + Adjective (e.g., "stomach cancer")
# ]
POS_PATTERNS = [
    ['NOUN'],                # Single noun (e.g., "liver")
    ['ADJ', 'NOUN'],         # Adjective + Noun (e.g., "small intestine")
    ['NOUN', 'NOUN'],        # Noun + Noun (e.g., "liver disease")
    ['ADJ', 'NOUN', 'NOUN'], # Adjective + Noun + Noun (e.g., "large intestine disease")
    ['NOUN', 'ADP', 'NOUN'], # Noun + Preposition + Noun (e.g., "stomach in body")
    ['NOUN', 'CCONJ', 'NOUN'], # Noun + Conjunction + Noun (e.g., "liver and pancreas")
    ['ADJ', 'ADJ', 'NOUN'],  # Adjective + Adjective + Noun (e.g., "big white brain")
    ['NOUN', 'ADJ'],          # Noun + Adjective (e.g., "stomach cancer")
    ['ADJ', 'NOUN', 'ADP', 'NOUN'], # Adjective + Noun + Preposition + Noun (e.g., "pancreas in body")
    ['NOUN', 'PRP$', 'NOUN'],  # Noun + Possessive Pronoun + Noun (e.g., "liver's function")
    ['NOUN', 'IN', 'NOUN'],   # Noun + Preposition + Noun (e.g., "intestines in stomach")
]


# ---------- Preprocessing ----------
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()

# ---------- POS Tagging ----------
def get_lemmas_and_pos(text, pos_tagger):
    tagged = pos_tagger(text)
    words = [t['word'].lower() for t in tagged]
    pos_tags = [t['entity_group'] for t in tagged]
    
    # Remove subwords (tokens starting with '##')
    words = [word for word in words if not word.startswith('##')]
    
    return words, pos_tags

# ---------- Match POS Patterns ----------
def match_patterns(words, pos_tags, patterns):
    matched = []
    for i in range(len(words)):
        # Skip tokens that are subwords (starting with '##')
        if words[i].startswith('##'):
            continue
        
        for pattern in patterns:
            window = len(pattern)
            if i + window <= len(pos_tags) and pos_tags[i:i + window] == pattern:
                phrase = " ".join(words[i:i + window])
                matched.append((phrase, pattern))
    return matched

# ---------- Unithood Score ----------
def unithood_score(phrase, doc_freqs):
    score = 0
    freq = doc_freqs[phrase]
    if len(phrase.split()) > 1:
        score += 1
    if freq >= 2:
        score += 1
    if re.search(r"\b[a-z]+\s[a-z]+", phrase):
        score += 1
    return score

# ---------- Termhood Score ----------
def termhood_score(phrase):
    score = 0
    stop_words = {"the", "and", "in", "of", "no", "on", "for", "a", "with", "to"}
    tokens = phrase.split()
    
    # Increase score for multi-word terms
    if len(tokens) > 1:
        score += 1
    
    # Decrease score if any of the words are common stop words
    if any(w in stop_words for w in tokens):
        score -= 1
    
    # Increase score for medical suffixes or common terms
    if any(re.search(r"(itis|oma|algia|emia|osis|pathy|plasty|ectomy|uria)", t) for t in tokens):
        score += 1
        
    return score


# ---------- Filter Irrelevant Terms ----------
def filter_irrelevant_terms(extracted_terms):
    irrelevant_terms = ['are', 'is', 'the', 'of', 'and', 'which']
    filtered_terms = [term for term in extracted_terms if term.lower() not in irrelevant_terms]
    return filtered_terms

# ---------- Main Term Extraction Function ----------
def extract_terms_verbose(emr_texts, threshold_u=0.5, threshold_t=0.5):
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
    start_time = time.time()
    predicted_terms = extract_terms_verbose(emr_texts)

    predicted_set = set(predicted_terms)
    ground_truth_set = set(ground_truth)

    true_positives = predicted_set & ground_truth_set
    false_positives = predicted_set - ground_truth_set
    false_negatives = ground_truth_set - predicted_set

    precision = len(true_positives) / (len(predicted_set) or 1)
    recall = len(true_positives) / (len(ground_truth_set) or 1)
    f1 = 2 * precision * recall / (precision + recall or 1)

    print("\n🔹 Evaluation Metrics")
    print(f"True Positives : {len(true_positives)} => {true_positives}")
    print(f"False Positives: {len(false_positives)} => {false_positives}")
    print(f"False Negatives: {len(false_negatives)} => {false_negatives}")
    print(f"\nPrecision: {precision:.2f}")
    print(f"Recall   : {recall:.2f}")
    print(f"F1 Score : {f1:.2f}")
    print(f"⏱️  Time Taken: {time.time() - start_time:.2f} seconds")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": list(true_positives),
        "false_positives": list(false_positives),
        "false_negatives": list(false_negatives)
    }

# ---------- Example Usage ----------
if __name__ == "__main__":
    emr_texts = [
        "The head contains the cranial cavity which is formed by the skull and encloses the brain.",
        "The trunk is composed of the thoracic, abdominal and pelvic cavities.",
        "The thoracic cavity is formed by sternum, ribs and thoracic vertebrae.",
        "The floor is formed by the diaphragm.",
        "The organs or viscera in the thoracic cavity are the heart, lungs, trachea and oesophagus.",
        "The abdominal cavity is formed by the vertebral column, and layers of muscle which support the viscera.",
        "The viscera in the abdominal cavity are the stomach, small and large intestines, liver, gallbladder, spleen, pancreas, and kidneys.",
        "The pelvic cavity is enclosed by the bony pelvis.",
        "The viscera in the pelvic cavity are the urinary bladder, organs of reproduction, sigmoid colon and rectum.",
        "The spinal canal is continuous with the cranial cavity and lies within the backbone.",
        "It encloses the spinal cord."
    ]

    ground_truth_terms = [
        "cranial cavity", "thoracic cavity", "abdominal cavity", "pelvic cavity", 
        "spinal canal", "vertebral column", "thoracic vertebrae", "viscera", 
        "stomach", "small intestines", "large intestines", "liver", "gallbladder", 
        "spleen", "pancreas", "kidneys", "urinary bladder", "sigmoid colon", 
        "rectum", "trachea", "oesophagus", "spinal cord", "brain", "diaphragm"
    ]

    evaluate_extraction(emr_texts, ground_truth_terms)
