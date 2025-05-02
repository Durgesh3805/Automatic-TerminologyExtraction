import nltk
from nltk import pos_tag, ngrams
from nltk.tokenize import TreebankWordTokenizer
from collections import defaultdict, Counter
import math
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from nltk.corpus import stopwords

# Download required NLTK resources
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords')

POS_PATTERNS = [['JJ', 'NN'], ['NN', 'NN'], ['NN', 'IN', 'NN'], ['NN', 'CC', 'NN'], ['NNP', 'NN']]
tokenizer = TreebankWordTokenizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    return text.lower()

def match_patterns(tokens, pos_tags):
    matches = []
    examples = []
    for size in [2, 3]:
        for i in range(len(pos_tags) - size + 1):
            pattern = [tag for word, tag in pos_tags[i:i+size]]
            if pattern in POS_PATTERNS:
                phrase = ' '.join([word for word, tag in pos_tags[i:i+size]])
                matches.append(phrase)
                examples.append((phrase, ' '.join(pattern)))
    return matches, examples

def generate_ngrams(tokens, n):
    return [' '.join(gram) for gram in ngrams(tokens, n)]

def surrounding_words(phrase, tokens, window=2):
    words = phrase.split()
    for i in range(len(tokens) - len(words) + 1):
        if tokens[i:i+len(words)] == words:
            start = max(0, i - window)
            end = min(len(tokens), i + len(words) + window)
            return tokens[start:i] + tokens[i+len(words):end]
    return []

def compute_cvalue(term, freq, nested_terms):
    if len(term.split()) < 2:
        return freq[term]
    nested_freqs = [freq[nt] for nt in nested_terms if nt in freq]
    mean_nested = sum(nested_freqs) / len(nested_freqs) if nested_freqs else 0
    return math.log2(len(term.split())) * (freq[term] - mean_nested)

def context_weights(context_words, known_medical_terms):
    return sum(1 for word in context_words if word in known_medical_terms)

def pointwise_mutual_info(term, total_terms, term_freq, cooc_freq):
    if len(term.split()) != 2:
        return 0
    w1, w2 = term.split()
    p_w1 = term_freq[w1] / total_terms
    p_w2 = term_freq[w2] / total_terms
    p_w1w2 = cooc_freq[term] / total_terms
    if p_w1w2 == 0 or p_w1 == 0 or p_w2 == 0:
        return 0
    return math.log2(p_w1w2 / (p_w1 * p_w2))

def is_valid_term(term):
    words = term.split()
    # Remove if all are stopwords or punctuation
    if all(w in stop_words or w in string.punctuation for w in words):
        return False
    # Remove single stopword terms
    if len(words) == 1 and words[0] in stop_words:
        return False
    return True

def extract_terms_verbose(emr_texts, alpha=0.4, beta=0.3, gamma=0.3, top_n=20):
    candidate_terms = defaultdict(float)
    freq = Counter()
    doc_map = defaultdict(set)
    context_info = defaultdict(list)
    total_docs = len(emr_texts)
    cooc_freq = Counter()
    term_freq = Counter()
    known_medical_terms = {'pain', 'fever', 'cancer', 'diabetes', 'infection'}

    all_docs_tokens = []
    matched_examples = []
    all_preprocessed = []

    for doc_id, text in enumerate(emr_texts):
        cleaned = preprocess(text)
        tokens = tokenizer.tokenize(cleaned)
        all_docs_tokens.append(tokens)
        all_preprocessed.append(' '.join(tokens))
        tagged = pos_tag(tokens)

        patterns, examples = match_patterns(tokens, tagged)
        matched_examples.extend(examples)

        for n in [1, 2, 3]:
            patterns += generate_ngrams(tokens, n)

        for phrase in patterns:
            freq[phrase] += 1
            doc_map[phrase].add(doc_id)
            context_info[phrase].extend(surrounding_words(phrase, tokens))
            for word in phrase.split():
                term_freq[word] += 1
            if len(phrase.split()) == 2:
                cooc_freq[phrase] += 1

    total_terms = sum(term_freq.values())

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([' '.join(tokens) for tokens in all_docs_tokens])
    feature_names = tfidf_vectorizer.get_feature_names_out()
    tfidf_scores = dict(zip(feature_names, tfidf_matrix.sum(axis=0).A1))

    scoring_table = []

    for phrase in freq:
        tf = freq[phrase]
        df = len(doc_map[phrase])
        idf = math.log((total_docs + 1) / (1 + df)) + 1
        tfidf = tf * idf

        nested_terms = [t for t in freq if t != phrase and t in phrase]
        cvalue = compute_cvalue(phrase, freq, nested_terms)

        context_score = context_weights(context_info[phrase], known_medical_terms)
        ncvalue = 0.8 * cvalue + 0.2 * context_score

        pmi = pointwise_mutual_info(phrase, total_terms, term_freq, cooc_freq)
        tscore = tf / math.sqrt(tf) if tf > 0 else 0
        chi_sq = (tf - df) ** 2 / df if df > 0 else 0
        combo = (pmi + tscore + chi_sq) / 3

        hybrid = alpha * ncvalue + beta * combo + gamma * tfidf
        candidate_terms[phrase] = hybrid

        unithood = round(combo, 2)
        termhood = round(ncvalue, 2)
        accepted = hybrid > 2
        scoring_table.append((phrase, unithood, termhood, "✅ Yes" if accepted else "❌ No"))

    sorted_terms = sorted(candidate_terms.items(), key=lambda x: x[1], reverse=True)
    
    # ✔️ Final filtering to remove non-informative terms
    final_terms = [term for term, _ in sorted_terms[:top_n] if is_valid_term(term)]

    # Verbose Output
    print("\n🔹 Preprocessing Output (Simplified)")
    for line in all_preprocessed:
        print(line)

    print("\n🔹 Matched PoS Patterns (Examples)")
    for phrase, pattern in matched_examples[:10]:
        print(f"{phrase} ({pattern})")

    print("\n🔹 Scoring (Hypothetical)")
    print(f"{'Phrase':<25} {'Unithood Score':<15} {'Termhood Score':<15} {'Accepted?'}")
    for phrase, u_score, t_score, acc in scoring_table[:15]:
        print(f"{phrase:<25} {u_score:<15} {t_score:<15} {acc}")

    print("\n✅ Final Output: Extracted Terms\n")
    print("[")
    for term in final_terms:
        print(f'    "{term}",')
    print("]")

    return final_terms

# Example use
documents = [
    "The patient exhibited acute renal failure and severe lung lesion.",
    "He has a history of diabetic nephropathy and chronic hypertension.",
    "CT scan revealed carcinoma in the upper lobe.",
    "Patient reports no infection but shows early signs of heart disease."
]

extract_terms_verbose(documents)
