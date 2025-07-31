import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("C:/Users/Dell/Downloads/In House/train.csv")

# Split data
X = df["text"]
y = df["labels"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    max_features=5000,  # You can adjust this
    ngram_range=(1,2),  # unigrams and bigrams
    stop_words="english"
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train SVM classifier
svm = LinearSVC()
svm.fit(X_train_tfidf, y_train)

# Predictions
y_pred = svm.predict(X_test_tfidf)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")

# Classification report
report = classification_report(y_test, y_pred, target_names=["No Effect", "Side Effect"])
print("\nClassification Report:\n")
print(report)

# Confusion matrix plot
disp = ConfusionMatrixDisplay.from_estimator(
    svm, X_test_tfidf, y_test, display_labels=["No Effect", "Side Effect"],
    cmap=plt.cm.Blues
)
plt.title("Confusion Matrix")
plt.show()

# Optional: Accuracy bar graph
plt.bar(["Accuracy"], [accuracy], color="green")
plt.ylim(0,1)
plt.ylabel("Accuracy")
plt.title("Model Accuracy")
plt.show()