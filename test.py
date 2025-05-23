import time
import math
from collections import defaultdict, Counter
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import precision_score, recall_score, f1_score

# Set up NLTK data
import nltk
nltk.data.path.append("C:/Users/USER-PC/AppData/Roaming/nltk_data")
nltk.download('averaged_perceptron_tagger_eng', download_dir="C:/Users/USER-PC/AppData/Roaming/nltk_data")

# Preprocessors
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    return text.lower()

def tokenize(text):
    return word_tokenize(text)

def nltk_pos_to_wordnet(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemmatize(tokens, pos_tags):
    return [lemmatizer.lemmatize(tok, nltk_pos_to_wordnet(pos)) for tok, pos in zip(tokens, pos_tags)]

def get_pos_tags(tokens):
    return pos_tag(tokens)

def match_patterns(lemmas, pos_tags, patterns):
    tags = [tag for _, tag in pos_tags]
    phrases = []
    for pattern in patterns:
        length = len(pattern)
        for i in range(len(tags) - length + 1):
            if [tags[i + j] for j in range(length)] == pattern:
                phrases.append(" ".join(lemmas[i:i + length]))
    return phrases

def generate_ngrams(tokens, n):
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def get_nested_terms(phrase):
    words = phrase.split()
    return [" ".join(words[i:j]) for i in range(len(words)) for j in range(i+1, len(words))]

def pointwise_mutual_information(phrase, freq, total_terms):
    words = phrase.split()
    if len(words) < 2:
        return 0
    p_phrase = freq[phrase] / total_terms
    p_individual = 1
    for word in words:
        p_individual *= freq.get(word, 1e-5) / total_terms
    return math.log2(p_phrase / p_individual) if p_individual > 0 else 0

def t_score(phrase, freq):
    return freq[phrase] / math.sqrt(freq[phrase]) if freq[phrase] > 0 else 0

def chi_squared_score(phrase, freq, total_terms):
    observed = freq[phrase]
    expected = total_terms / len(freq)
    return ((observed - expected) ** 2) / expected if expected > 0 else 0

def context_weights(context_words, known_terms_freq):
    return sum(known_terms_freq.get(w, 0) for w in context_words)

# Hybrid Terminology Extraction with CNC Combo Scores
def HybridTerminologyExtractionWithCNCComboScores(EMR_Texts, alpha=0.5, beta=0.3, gamma=0.2, top_n=50, threshold=0):
    pos_patterns = [
    ['JJ', 'NN'],           # Adjective + Noun
    ['NN', 'NN'],           # Noun + Noun (e.g., kidney disease)
    ['NN', 'IN', 'NN'],     # Noun + Preposition + Noun (e.g., pain in chest)
    ['NN', 'CC', 'NN'],     # Noun + Conjunction + Noun
    ['NNP', 'NN'],          # Proper Noun + Noun (e.g., disease names)
    ['JJ', 'NN', 'IN'],     # Adjective + Noun + Preposition (e.g., chronic kidney)
    ['VBG', 'NN'],          # Gerund + Noun (e.g., treating disease)
]

    freq = Counter()
    phrase_doc_map = defaultdict(set)
    context_info = defaultdict(list)
    total_documents = len(EMR_Texts)

    for doc_id, doc in enumerate(EMR_Texts):
        cleaned = preprocess(doc)
        tokens = tokenize(cleaned)
        pos_tags = get_pos_tags(tokens)
        lemmas = lemmatize(tokens, [tag for _, tag in pos_tags])

        phrases = match_patterns(lemmas, pos_tags, pos_patterns)
        for n in [1, 2, 3]:
            phrases += generate_ngrams(lemmas, n)

        for phrase in phrases:
            freq[phrase] += 1
            phrase_doc_map[phrase].add(doc_id)
            phrase_words = phrase.split()
            idx = [i for i, tok in enumerate(lemmas) if tok in phrase_words]
            surrounding = [lemmas[i] for i in range(max(0, min(idx)-3), min(len(lemmas), max(idx)+4))]
            context_info[phrase].extend(surrounding)

    total_terms = sum(freq.values())
    candidate_terms = {}

    for phrase in freq:
        tf = freq[phrase]
        df = len(phrase_doc_map[phrase])
        idf = math.log((total_documents + 1) / (1 + df))
        tfidf = tf * idf

        if len(phrase.split()) >= 2:
            nested = get_nested_terms(phrase)
            nested_freqs = [freq.get(t, 0) for t in nested]
            mean_nested = sum(nested_freqs) / len(nested_freqs) if nested_freqs else 0
            c_value = math.log2(len(phrase.split())) * (tf - mean_nested)
        else:
            c_value = tfidf

        context_weight = context_weights(context_info[phrase], freq)
        nc_value = 0.8 * c_value + 0.2 * context_weight

        pmi = pointwise_mutual_information(phrase, freq, total_terms)
        ts = t_score(phrase, freq)
        chi2 = chi_squared_score(phrase, freq, total_terms)
        combo_score = (pmi + ts + chi2) / 3

        hybrid_score = alpha * nc_value + beta * combo_score + gamma * tfidf
        if hybrid_score > threshold:
            candidate_terms[phrase] = hybrid_score

    sorted_terms = sorted(candidate_terms.items(), key=lambda x: x[1], reverse=True)
    extracted_terms = [term for term, _ in sorted_terms[:top_n]]

    return extracted_terms

# Example usage
emr_docs = [
    "The patient had chronic kidney disease and elevated blood pressure.",
    "Diabetes mellitus type 2 was diagnosed with high glucose levels.",
    "Patient reports acute chest pain and shortness of breath."
]

# Start timer
start_time = time.time()

# Run the terminology extraction
terms = HybridTerminologyExtractionWithCNCComboScores(emr_docs)

# Stop timer
end_time = time.time()
exec_time = end_time - start_time

# Print evaluation results
ground_truth = set([
    "chronic kidney disease",
    "blood pressure",
    "diabetes mellitus",
    "glucose levels",
    "chest pain",
    "shortness of breath"
])

# Convert to sets for evaluation
predicted = set(terms)
true_positives = predicted & ground_truth
false_positives = predicted - ground_truth
false_negatives = ground_truth - predicted

# Evaluation metrics
tp = len(true_positives)
fp = len(false_positives)
fn = len(false_negatives)

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print("\n🔍 Evaluation Results")
print(f"True Positives : {tp}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"Precision      : {precision:.4f}")
print(f"Recall         : {recall:.4f}")
print(f"F1-Score       : {f1:.4f}")
print(f"Execution Time : {exec_time:.2f} seconds")
