import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from utils import clean_text

def train_model():

    print("Loading Dataset...")
    df = pd.read_excel("fake_job_postings.xlsx.xlsx")

   # change to read_excel if needed

    # Combine important text columns
    df['text'] = df['title'].fillna('') + " " + df['description'].fillna('')

    print("Cleaning Text...")
    df['text'] = df['text'].apply(clean_text)

    X = df['text']
    y = df['fraudulent']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training Model...")
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)

    # Save model and vectorizer
    pickle.dump(model, open("model.pkl", "wb"))
    pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

    print("Model Saved Successfully!")

if __name__ == "__main__":
    train_model()
