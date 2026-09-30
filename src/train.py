import sys
import glob
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, accuracy_score)
from preprocess import load_data


def find_dataset() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    files = glob.glob("data/*.csv") + glob.glob("*.csv")
    if not files:
        sys.exit("No CSV found. Put the dataset in data/ or pass its path.")
    print(f"Using dataset: {files[0]}")
    return files[0]


df = load_data(find_dataset())
print(f"Samples: {len(df)} | Spam ratio: {df['label'].mean():.2%}")

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

models = {
    "naive_bayes": MultinomialNB(alpha=0.1),
    "svm": LinearSVC(C=1.0, class_weight="balanced"),
}

best_name, best_f1, best_pipe, best_cm = None, -1, None, None

for name, clf in models.items():
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                  min_df=2, max_features=50000)),
        ("clf", clf),
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    f1 = f1_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    print(f"\n=== {name} ===")
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(classification_report(y_test, preds, target_names=["ham", "spam"]))
    print("Confusion matrix:\n", cm)

    if f1 > best_f1:
        best_name, best_f1, best_pipe, best_cm = name, f1, pipe, cm

print(f"\nBest model: {best_name} (spam F1 = {best_f1:.4f})")
joblib.dump(best_pipe, "models/spam_model.joblib")
print("Saved: models/spam_model.joblib")

# Optional: save confusion matrix image for the README
try:
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    ConfusionMatrixDisplay(best_cm, display_labels=["ham", "spam"]).plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {best_name}")
    plt.savefig("models/confusion_matrix.png", dpi=120, bbox_inches="tight")
    print("Saved: models/confusion_matrix.png")
except ImportError:
    pass