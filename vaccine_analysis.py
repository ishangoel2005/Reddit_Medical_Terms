import pandas as pd
import numpy as np
import spacy
import scispacy
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter

# Load medical NER model
nlp = spacy.load("en_ner_bc5cdr_md")

# Load the dataset
df = pd.read_csv("C:/Users/Dell/Downloads/In House/train.csv")

# Separate texts by labels
true_texts = df[df['labels'] == 1]['text']
no_effect_texts = df[df['labels'] == 0]['text']

def extract_medical_entities(text_series):
    all_terms = []
    for doc in nlp.pipe(text_series, disable=["tagger", "parser"]):
        for ent in doc.ents:
            if ent.label_ in ["DISEASE", "CHEMICAL"]:
                all_terms.append(ent.text.lower())
    return all_terms

# Extract medical terms for each label
true_medical_terms = extract_medical_entities(true_texts)
no_effect_medical_terms = extract_medical_entities(no_effect_texts)

# Count frequencies
true_freq = Counter(true_medical_terms)
no_effect_freq = Counter(no_effect_medical_terms)

# Convert to DataFrames
top_true_df = pd.DataFrame(true_freq.most_common(15), columns=['medical_term', 'count'])
top_true_df['label'] = '1 (Side Effect)'

top_no_effect_df = pd.DataFrame(no_effect_freq.most_common(15), columns=['medical_term', 'count'])
top_no_effect_df['label'] = '0 (No Effect)'

# Merge and save
final_medical_df = pd.concat([top_true_df, top_no_effect_df])
final_medical_df.to_csv("medical_keywords_summary.csv", index=False)

print("File saved: medical_keywords_summary.csv")
