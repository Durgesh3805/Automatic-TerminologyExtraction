import nltk
import networkx as nx
from nltk.corpus import stopwords
from itertools import combinations
import re
import time
from nltk.util import ngrams
from fuzzywuzzy import fuzz  # For fuzzy matching

# Download NLTK stopwords if not already
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))

# -------------------------------
# Step 1: Text Preprocessing
# -------------------------------
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove punctuation
    tokens = nltk.word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return tokens

# -------------------------------
# Step 2: N-gram Extraction (Improved)
# -------------------------------
def extract_ngrams(tokens, n=2):
    n_grams = ngrams(tokens, n)
    ngram_list = [' '.join(ngram) for ngram in n_grams]
    return ngram_list

# -------------------------------
# Step 3: Build Graph from Tokens and N-grams (Improved)
# -------------------------------
def build_graph(tokens, window_size=4):
    graph = nx.Graph()
    ngrams_list = extract_ngrams(tokens, 2)  # Extract bigrams for multi-word terms
    tokens.extend(ngrams_list)  # Add bigrams to token list for graph building
    
    # Create the graph by adding edges between adjacent tokens and bigrams
    for i in range(len(tokens)):
        for j in range(i + 1, i + window_size):
            if j >= len(tokens):
                break
            w1, w2 = tokens[i], tokens[j]
            if w1 != w2:
                graph.add_edge(w1, w2, weight=graph[w1][w2]['weight'] + 1 if graph.has_edge(w1, w2) else 1)
    return graph

# -------------------------------
# Step 4: Apply PageRank
# -------------------------------
def textrank_scores(graph):
    return nx.pagerank(graph, weight='weight')

# -------------------------------
# Step 5: Extract Top Terms
# -------------------------------
def extract_keywords(text, top_k=10):
    tokens = preprocess_text(text)
    graph = build_graph(tokens)
    scores = textrank_scores(graph)
    sorted_terms = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [term for term, score in sorted_terms[:top_k]]

# -------------------------------
# Evaluate Extraction (with Fuzzy Matching)
# -------------------------------
def evaluate_extraction(predicted_terms, ground_truth):
    """
    Evaluate predicted terms against ground truth.
    Calculates Precision, Recall, F1-score, and execution time.
    """
    start_time = time.time()

    pred_set = set(predicted_terms)
    true_set = set(item for sublist in ground_truth for item in sublist)

    # Fuzzy matching for multi-word terms
    tp, fp, fn = 0, 0, 0
    for true_term in true_set:
        matched = False
        for pred_term in pred_set:
            if fuzz.partial_ratio(true_term, pred_term) > 80:  # If the match is above a threshold
                tp += 1
                matched = True
                break
        if not matched:
            fn += 1

    for pred_term in pred_set:
        matched = False
        for true_term in true_set:
            if fuzz.partial_ratio(true_term, pred_term) > 80:
                matched = True
                break
        if not matched:
            fp += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    end_time = time.time()
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
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "exec_time": exec_time
    }

# -------------------------------
# Example Use
# -------------------------------
if __name__ == "__main__":
    clinical_text = """
    The patient has a history of hypertension and diabetes. She complains of chest pain,
    shortness of breath, and fatigue. Vital signs are stable. ECG shows sinus tachycardia.
    """
    
    # Define ground truth terms (for evaluation purposes)
    ground_truth = [
        ['hypertension', 'diabetes', 'chest pain', 'shortness of breath', 'fatigue', 'ecg', 'tachycardia'],
    ]

    # Extract keywords from the text
    keywords = extract_keywords(clinical_text, top_k=10)
    print("Extracted EMR Terms:", keywords)

    # Evaluate the extraction using the ground truth
    evaluate_extraction(keywords, ground_truth)
