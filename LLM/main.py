import random

import spacy
from spacy.util import minibatch
from spacy.training.example import Example

train_data = [
    ("What is the price of 10 bananas?", {"entities": [(21, 23, "QUANTITY"), (24, 31, "PRODUCT")]}),
    ("What is the price of 5 laptops?", {"entities": [(21, 22, "QUANTITY"), (23, 30, "PRODUCT")]}),
    ("How much are 7 bottles?", {"entities": [(13, 14, "QUANTITY"), (15, 22, "PRODUCT")]}),
    ("Could I buy 17 phones from you?", {"entities": [(12, 14, "QUANTITY"), (15, 21, "PRODUCT")]}),
    ("I am interested in acquiring 10 books.", {"entities": [(31, 33, "QUANTITY"), (34, 39, "PRODUCT")]}),
    ("Can you get me 12 apples?", {"entities": [(16, 18, "QUANTITY"), (19, 25, "PRODUCT")]}),
    ("Please check the price of 3 pens.", {"entities": [(26, 27, "QUANTITY"), (28, 32, "PRODUCT")]}),
    ("What do 5 chairs cost?", {"entities": [(8, 9, "QUANTITY"), (10, 16, "PRODUCT")]}),
    ("I need 8 laptops for the office.", {"entities": [(7, 8, "QUANTITY"), (9, 16, "PRODUCT")]}),
    ("How much for 4 desks?", {"entities": [(13, 14, "QUANTITY"), (15, 20, "PRODUCT")]}),
    ("We require 20 notebooks.", {"entities": [(11, 13, "QUANTITY"), (14, 23, "PRODUCT")]}),
    ("I want to order 6 cameras.", {"entities": [(17, 18, "QUANTITY"), (19, 26, "PRODUCT")]}),
    ("Can you provide the price of 2 watches?", {"entities": [(28, 29, "QUANTITY"), (30, 37, "PRODUCT")]}),
    ("Tell me the cost of 3 bottles.", {"entities": [(20, 21, "QUANTITY"), (22, 29, "PRODUCT")]}),
    ("Is it possible to buy 7 microphones?", {"entities": [(24, 25, "QUANTITY"), (26, 37, "PRODUCT")]}),
    ("I would like 10 keyboards.", {"entities": [(14, 16, "QUANTITY"), (17, 26, "PRODUCT")]}),
    ("Give me 15 calculators.", {"entities": [(8, 10, "QUANTITY"), (11, 22, "PRODUCT")]}),
    ("Could I get 9 mouse pads?", {"entities": [(12, 13, "QUANTITY"), (14, 24, "PRODUCT")]}),
    ("Can you tell me about 6 monitors?", {"entities": [(20, 21, "QUANTITY"), (22, 30, "PRODUCT")]}),
    ("I'm looking for 11 hard drives.", {"entities": [(16, 18, "QUANTITY"), (19, 30, "PRODUCT")]}),
    ("Do you have 5 power banks?", {"entities": [(12, 13, "QUANTITY"), (14, 25, "PRODUCT")]}),
    ("How about 14 USB drives?", {"entities": [(10, 12, "QUANTITY"), (13, 23, "PRODUCT")]}),
    ("We need 18 projectors for the event.", {"entities": [(8, 10, "QUANTITY"), (11, 21, "PRODUCT")]}),
    ("Could you send 4 printers to the office?", {"entities": [(13, 14, "QUANTITY"), (15, 23, "PRODUCT")]}),
    ("Are 3 scanners available?", {"entities": [(4, 5, "QUANTITY"), (6, 14, "PRODUCT")]}),
]

nlp = spacy.blank("en")

if 'ner' not in nlp.pipe_names:
    ner = nlp.add_pipe('ner')
else:
    ner = nlp.get_pipe('ner')

for _, annotations in train_data:
    for ent in annotations.get('entities', []):
        if ent[2] not in ner.labels:
            ner.add_label(ent[2])

other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.begin_training()

    epochs = 50
    for epoch in range(epochs):
        random.shuffle(train_data)
        losses = {}
        batches = minibatch(train_data, size=2)
        for batch in batches:
            examples = []
            for text, annotations in batch:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                examples.append(example)
            nlp.update(examples, drop=0.5, losses=losses)
        print(f'Epoch {epoch + 1}, Losses: {losses}')

nlp.to_disk('custom_ner_model')

trained_nlp = spacy.load('custom_ner_model')

text_texts = [
    "How much for 3 oranges?",
    "I need 15 chairs for the conference.",
    "Can you give me the price for 6 desks?",
]

for text in text_texts:
    doc = trained_nlp(text)
    print(f'Text: {text}')
    print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
    print()