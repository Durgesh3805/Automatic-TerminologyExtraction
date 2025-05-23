import re
import time
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources if not done already
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')
# nltk.download('punkt')
nltk.download('wordnet')

# ---------- Preprocessing Function with Lemmatization and Custom Stopwords ----------
def preprocess(text, custom_stopwords):
    """
    Clean and preprocess the text by converting it to lowercase, removing non-alphabetical characters,
    and lemmatizing the words.
    """
    # Lemmatizer and stopwords
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)  # Remove non-alphabetic characters
    
    # Tokenize and lemmatize
    words = nltk.word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words if word not in custom_stopwords]
    
    return " ".join(lemmatized_words)

# ---------- TF-IDF Extraction with N-grams and Adjusted Parameters ----------
def extract_terms_statistical(emr_texts, threshold=0.2, ngram_range=(1, 2)):
    """
    Extract terms from a list of EMR (Electronic Medical Record) texts using a statistical approach (TF-IDF).
    """
    # Define custom stopwords (can be extended)
    custom_stopwords = set(stopwords.words('english'))

    # Preprocess texts with custom stopwords
    cleaned_texts = [preprocess(text, custom_stopwords) for text in emr_texts]

    # TF-IDF Vectorizer with N-grams and adjusted min_df/max_df
    # vectorizer = TfidfVectorizer(max_df=0.85, min_df=0.01, ngram_range=ngram_range, stop_words='english')
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=0.02, ngram_range=(1, 2), stop_words='english')

    X = vectorizer.fit_transform(cleaned_texts)

    # Get feature names (terms)
    feature_names = vectorizer.get_feature_names_out()

    # Extract the terms with the highest TF-IDF scores
    scores = X.sum(axis=0).A1  # Sum TF-IDF scores across all documents
    term_scores = [(term, score) for term, score in zip(feature_names, scores)]

    # Sort terms by score in descending order
    term_scores_sorted = sorted(term_scores, key=lambda x: x[1], reverse=True)

    # Filter terms based on threshold
    extracted_terms = [term for term, score in term_scores_sorted if score >= threshold]
   

    
    print("\n🔹 Extracted Terms (TF-IDF) based on threshold:", threshold)
    for term in extracted_terms:
        print(f"- {term}")

    return extracted_terms

# ---------- Evaluation Function ----------
def evaluate_extraction_statistical(emr_texts, ground_truth):
    """
    Evaluates the term extraction process by comparing predicted terms with ground truth data using statistical approach.
    """
    start_time = time.time()
    predicted_terms = extract_terms_statistical(emr_texts)
    end_time = time.time()

    # Convert lists to sets for comparison
    pred_set = set(predicted_terms)
    true_set = set([item for sublist in ground_truth for item in sublist])

    # Calculate precision, recall, and F1-score
    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1)
    exec_time = end_time - start_time

    print("\n🔍 Evaluation Results (TF-IDF)")
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
    evaluate_extraction_statistical(emr_texts, ground_truth)
