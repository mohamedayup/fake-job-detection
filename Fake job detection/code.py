import tkinter as tk
import pickle
from utils import clean_text

# Load saved model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def predict():
    user_input = text_box.get("1.0", tk.END)
    cleaned = clean_text(user_input)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]

    if prediction == 1:
        result_label.config(text="❌ FAKE Job Posting", fg="red")
    else:
        result_label.config(text="✅ REAL Job Posting", fg="green")

# GUI Window
window = tk.Tk()
window.title("Fake Job Detection System")
window.geometry("600x400")

title_label = tk.Label(window, text="Enter Job Description",
                       font=("Arial", 16))
title_label.pack(pady=10)

text_box = tk.Text(window, height=10, width=70)
text_box.pack()

predict_button = tk.Button(window, text="Check Job",
                           command=predict,
                           font=("Arial", 14))
predict_button.pack(pady=10)

result_label = tk.Label(window, text="", font=("Arial", 16))
result_label.pack(pady=10)

window.mainloop()
