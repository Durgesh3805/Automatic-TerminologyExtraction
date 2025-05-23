from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import re
import nltk
from nltk.corpus import stopwords
from collections import Counter
import time
from sklearn.metrics import precision_recall_fscore_support

# Initialize domain-specific model for POS tagging (e.g., ClinicalBERT)
model_name = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
pos_tagger = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Preprocess with extended stopwords and domain knowledge
nltk.download('stopwords')
custom_stopwords = set(stopwords.words('english')).union({"condition", "treatment", "patient"})

def preprocess(text, custom_stopwords):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)  # Remove non-alphabetic characters
    words = nltk.word_tokenize(text)
    lemmatized_words = [nltk.WordNetLemmatizer().lemmatize(word) for word in words if word not in custom_stopwords]
    return " ".join(lemmatized_words)

# POS Tagging function
def get_lemmas_and_pos(text, pos_tagger):
    tagged = pos_tagger(text)
    words = [t['word'].lower() for t in tagged]
    pos_tags = [t['entity_group'] for t in tagged]
    return words, pos_tags

# Improved scoring functions
def unithood_score(phrase, doc_freqs):
    score = 0
    freq = doc_freqs[phrase]
    if len(phrase.split()) > 1:  # Multiple words
        score += 1
    if freq >= 2:  # Frequent terms
        score += 1
    if any(re.search(r"(itis|oma|algia|emia|osis|pathy|plasty|ectomy)", word) for word in phrase.split()):
        score += 1
    return score

# Extraction process
def extract_terms_statistical(emr_texts, threshold_u=0.5, threshold_t=0.5):
    all_phrases = []
    all_matches = []
    print("\n🔹 Preprocessing Output\n")
    for text in emr_texts:
        print(f"Original: {text}")
        clean_text = preprocess(text, custom_stopwords)
        words, pos_tags = get_lemmas_and_pos(clean_text, pos_tagger)
        phrases = match_patterns(words, pos_tags, POS_PATTERNS)
        simplified = sorted(set([p[0] for p in phrases]))
        all_phrases.extend(simplified)
        all_matches.extend(phrases)
        for s in simplified:
            print(f"- {s}")
        print()

    doc_freqs = Counter(all_phrases)

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

# Example evaluation function
def evaluate_extraction(emr_texts, ground_truth):
    start_time = time.time()
    predicted_terms = extract_terms_statistical(emr_texts)
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

# Example Usage
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
