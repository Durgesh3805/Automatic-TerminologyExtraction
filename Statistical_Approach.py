import nltk
import string
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Sample EMR texts
EMR_Texts = [
    "The patient exhibited acute renal failure and severe lung lesion.",
    "He has a history of diabetic nephropathy and chronic hypertension.",
    "CT scan revealed carcinoma in the upper lobe.",
    "Patient reports no infection but shows early signs of heart disease."
]

# Tokenizer with fix for punkt_tab issue
def tokenize(text):
    return word_tokenize(text, preserve_line=True)  # avoid sentence split

# Convert POS tag to WordNet format
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    return wordnet.NOUN

# Preprocessing: lowercase, remove punctuation/stopwords, lemmatize
def preprocess(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = tokenize(text)
    tagged = pos_tag(tokens)
    lemmatized = [
        lemmatizer.lemmatize(word, get_wordnet_pos(pos))
        for word, pos in tagged
        if word not in stop_words
    ]
    return lemmatized

# Extract noun phrases based on POS patterns
def extract_phrases(tokens):
    phrases = []
    for i in range(len(tokens) - 1):
        bigram = tokens[i:i+2]
        pos = pos_tag(bigram)
        tags = [tag for _, tag in pos]
        if tags in [['JJ', 'NN'], ['NN', 'NN'], ['NNP', 'NN']]:
            phrases.append(" ".join(bigram))
    if len(tokens) >= 3:
        for i in range(len(tokens) - 2):
            trigram = tokens[i:i+3]
            pos = pos_tag(trigram)
            tags = [tag for _, tag in pos]
            if tags == ['JJ', 'NN', 'NN']:
                phrases.append(" ".join(trigram))
    return phrases

# Dummy scoring functions (replace with real ones if available)
def UnithoodScore(phrase):
    return phrase.count(' ') + 1  # crude example: 2-word phrase = score 2

def TermhoodScore(phrase):
    return phrase.count(' ') + 1  # crude example

# Main function with verbose output
def StatisticalTerminologyExtractionWithVerboseOutput(texts, threshold_u=2, threshold_t=2):
    all_phrases = set()
    print("\n🔹 Preprocessing Output (Simplified)")
    for text in texts:
        tokens = preprocess(text)
        print(" ".join(tokens))
        all_phrases.update(extract_phrases(tokens))

    print("\n🔹 Matched PoS Patterns (Examples)")
    for phrase in all_phrases:
        pos_tags = [tag for _, tag in pos_tag(phrase.split())]
        print(f"{phrase} ({' '.join(pos_tags)})")

    print("\n\n🔹 Scoring (Hypothetical)")
    print(f"{'Phrase':<25}{'Unithood Score':<15}{'Termhood Score':<15}{'Accepted?'}")
    accepted_terms = []
    for phrase in all_phrases:
        u_score = UnithoodScore(phrase)
        t_score = TermhoodScore(phrase)
        accepted = u_score >= threshold_u and t_score >= threshold_t
        print(f"{phrase:<25}{u_score:<15}{t_score:<15}{'✅ Yes' if accepted else '❌ No'}")
        if accepted:
            accepted_terms.append(phrase)

    print("\n✅ Final Output: Extracted Terms\n")
    print("[")
    for term in accepted_terms:
        print(f'    "{term}",')
    print("]")

# Run it
if __name__ == "__main__":
    StatisticalTerminologyExtractionWithVerboseOutput(EMR_Texts)
