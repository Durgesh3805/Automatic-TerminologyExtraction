import time
from transformers import pipeline
from sklearn.metrics import precision_score, recall_score, f1_score

# Load NER model
model_name = "d4data/biomedical-ner-all"
ner_pipeline = pipeline("ner", model=model_name, tokenizer=model_name, aggregation_strategy="simple")

# Input text and ground truth
clinical_text = """
The patient has a history of hypertension and diabetes. She complains of chest pain,
shortness of breath, and fatigue. Vital signs are stable. ECG shows sinus tachycardia.
"""

# ✅ Define ground truth entities (example)
true_entities = {
    "Hypertension", "Diabetes", "Chest pain", "Shortness of breath",
    "Fatigue", "Vital signs", "ECG", "Sinus tachycardia"
}

# Run NER
start = time.time()
ner_results = ner_pipeline(clinical_text)
exec_time = time.time() - start

# Filter predicted entities
def clean_entities(results):
    entities = set()
    for entity in results:
        word = entity['word'].replace("##", "").strip().lower()
        if len(word) > 2 and any(v in word for v in "aeiou"):
            entities.add(word.capitalize())
    return entities

predicted_entities = clean_entities(ner_results)

# Normalize both for comparison
pred_norm = set(e.lower() for e in predicted_entities)
true_norm = set(e.lower() for e in true_entities)

# Calculate TP, FP, FN
tp = len(pred_norm & true_norm)
fp = len(pred_norm - true_norm)
fn = len(true_norm - pred_norm)

# Avoid division by zero
precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

# ✅ Print entities
print("Extracted Medical Entities:")
for ent in predicted_entities:
    print(ent)

# ✅ Print evaluation results
print("\n🔍 Evaluation Results")
print(f"True Positives : {tp}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"Precision      : {precision:.4f}")
print(f"Recall         : {recall:.4f}")
print(f"F1-Score       : {f1:.4f}")
print(f"Execution Time : {exec_time:.2f} seconds")
