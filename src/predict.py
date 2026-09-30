import sys
import joblib
from preprocess import clean_text

model = joblib.load("models/spam_model.joblib")


def predict(text: str) -> str:
    return "SPAM" if model.predict([clean_text(text)])[0] == 1 else "HAM"


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(predict(" ".join(sys.argv[1:])))
    else:
        while True:
            text = input("\nEmail text (or 'q' to quit): ")
            if text.strip().lower() == "q":
                break
            print("->", predict(text))