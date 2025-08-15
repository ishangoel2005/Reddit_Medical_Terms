import spacy

# text = [
#     'John goes for a walk in Berlin.',
#     'Mike is going to the store',
#     'Elon Musk is the CEO of Twitter.',
#     'Bob Smith is the guy behind XYZ-SOft Inc.',
#     'Florian Dedov is the guy behind NeuralNine.'
# ]

text = [
    'What is the price of 4 bananas?',
    'How much are 16 chairs?',
    'Give me the value of 5 laptops.',
]

nlp = spacy.load("en_core_web_md")

# ner_labels = nlp.get_pipe('ner').labels
# print(ner_labels)

categories = ['PERSON', 'CARDINAL', 'ORG', 'LOC']

docs = [nlp(text) for text in text]

for doc in docs:
    entities = []
    for ent in doc.ents:
        if ent.label_ in categories:
            entities.append((ent.text, ent.label_))

    print(entities)