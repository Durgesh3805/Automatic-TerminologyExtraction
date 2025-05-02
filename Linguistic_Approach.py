import re
from collections import Counter
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Load transformer model and tokenizer for PoS tagging
model_name = "vblagoje/bert-english-uncased-finetuned-pos"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
pos_tagger = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# POS patterns to match
pos_patterns = [
    ['ADJ', 'NOUN'],
    ['NOUN', 'NOUN'],
    ['ADJ', 'NOUN', 'NOUN'],
    ['PROPN', 'NOUN'],
    ['NOUN', 'ADP', 'NOUN'],
    ['ADJ', 'NOUN', 'NOUN']
]

# ---------- Helper Functions ----------

def get_lemmas_and_pos(text):
    tagged = pos_tagger(text)
    words = [t['word'].lower() for t in tagged]
    pos_tags = [t['entity_group'] for t in tagged]
    return words, pos_tags

def match_patterns(words, pos_tags, pos_patterns):
    matched = []
    for window in range(2, 4 + 1):  # 2 to 4-grams
        for i in range(len(words) - window + 1):
            sub_words = words[i:i + window]
            sub_tags = pos_tags[i:i + window]
            if sub_tags in pos_patterns:
                matched.append((" ".join(sub_words), sub_tags))
    return matched

def unithood_score(phrase, doc_freqs):
    score = 0
    if doc_freqs[phrase] >= 1:
        score += 1
    if len(phrase.split()) > 1:
        score += 1
    if re.search(r"\b[a-z]+\s[a-z]+", phrase):  # basic multi-word check
        score += 1
    return score

def termhood_score(phrase):
    score = 0
    if len(phrase.split()) > 1:
        score += 1
    if not any(w in {"the", "and", "in", "of", "no"} for w in phrase.split()):
        score += 1
    if any(char.isalpha() for char in phrase):
        score += 1
    return score

# ---------- Main Function ----------

def extract_terms_verbose(emr_texts, threshold_u=2, threshold_t=2):
    all_phrases = []
    all_matches = []

    print("\n🔹 Preprocessing Output (Simplified)\n")

    for text in emr_texts:
        print(f"Original: {text}")
        words, pos_tags = get_lemmas_and_pos(text)
        phrases = match_patterns(words, pos_tags, pos_patterns)
        simplified = sorted(set([p[0] for p in phrases]))
        all_phrases.extend(simplified)
        all_matches.extend(phrases)
        for s in simplified:
            print(f"- {s}")
        print()

    doc_freqs = Counter(all_phrases)

    print("\n🔹 Matched PoS Patterns (Examples)\n")
    for phrase, tags in all_matches:
        print(f"{phrase:<30} {tags}")

    print("\n\n🔹 Scoring (Unithood & Termhood)\n")
    results = []
    for phrase in sorted(set(all_phrases)):
        u_score = unithood_score(phrase, doc_freqs)
        t_score = termhood_score(phrase)
        accepted = u_score >= threshold_u and t_score >= threshold_t
        results.append((phrase, u_score, t_score, accepted))
        print(f"{phrase:<30} U: {u_score} | T: {t_score} | {'✅ Yes' if accepted else '❌ No'}")

    final_terms = [r[0] for r in results if r[3]]

    print("\n\n✅ Final Output: Extracted Terms")
    print(final_terms)

    return final_terms

# ---------- Example Usage ----------

if __name__ == "__main__":
    EMR_Texts = [
        "The patient exhibited acute renal failure and severe lung lesion.",
        "He has a history of diabetic nephropathy and chronic hypertension.",
        "CT scan revealed carcinoma in the upper lobe.",
        "Patient reports no infection but shows early signs of heart disease.",
        "Patient has chronic kidney disease and hypertension.",
        "History of diabetes and high blood pressure.",
        "The patient suffers from end stage renal failure."
    ]

    extract_terms_verbose(EMR_Texts)
