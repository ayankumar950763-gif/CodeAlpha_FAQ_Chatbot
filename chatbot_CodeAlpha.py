"""
CodeAlpha AI Internship - Task 2: Chatbot for FAQs
Pipeline: FAQs -> NLTK preprocessing -> TF-IDF vectors -> cosine similarity -> best answer
Run:  python chatbot.py          (terminal chat)
      python chatbot.py --gui    (simple Tkinter chat window)
"""
import json
import string
import sys
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

THRESHOLD = 0.25  # below this similarity, the bot says it doesn't understand


def setup_nltk():
    for pkg in ("punkt", "punkt_tab", "stopwords", "wordnet"):
        nltk.download(pkg, quiet=True)


class FAQChatbot:
    def __init__(self, faq_path="faqs.json"):
        setup_nltk()
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self.faqs = json.loads(Path(faq_path).read_text(encoding="utf-8"))
        self.questions = [f["question"] for f in self.faqs]
        self.vectorizer = TfidfVectorizer()
        self.faq_vectors = self.vectorizer.fit_transform(
            [self.preprocess(q) for q in self.questions]
        )

    def preprocess(self, text):
        """lowercase -> tokenize -> remove punctuation/stopwords -> lemmatize"""
        tokens = word_tokenize(text.lower())
        tokens = [t for t in tokens if t not in string.punctuation]
        tokens = [t for t in tokens if t not in self.stop_words]
        return " ".join(self.lemmatizer.lemmatize(t) for t in tokens)

    def get_response(self, user_input):
        cleaned = self.preprocess(user_input)
        if not cleaned.strip():
            return "Please type a question so I can help you."
        vec = self.vectorizer.transform([cleaned])
        scores = cosine_similarity(vec, self.faq_vectors)[0]
        best = scores.argmax()
        if scores[best] < THRESHOLD:
            return ("Sorry, I couldn't find an answer to that. Try rephrasing, "
                    "or contact support@example.com.")
        return self.faqs[best]["answer"]


def run_cli(bot):
    print("ShopBot: Hi! Ask me anything about orders, shipping, returns or payments.")
    print("         (type 'quit' to exit)\n")
    while True:
        user = input("You: ").strip()
        if user.lower() in {"quit", "exit", "bye"}:
            print("ShopBot: Goodbye! Have a great day.")
            break
        print(f"ShopBot: {bot.get_response(user)}\n")


def run_gui(bot):
    import tkinter as tk
    from tkinter import scrolledtext

    root = tk.Tk()
    root.title("ShopBot - FAQ Chatbot")
    chat = scrolledtext.ScrolledText(root, width=60, height=20, wrap=tk.WORD, state="disabled")
    chat.pack(padx=10, pady=10)
    entry = tk.Entry(root, width=50)
    entry.pack(side=tk.LEFT, padx=(10, 5), pady=(0, 10))

    def add(sender, msg):
        chat.config(state="normal")
        chat.insert(tk.END, f"{sender}: {msg}\n\n")
        chat.config(state="disabled")
        chat.yview(tk.END)

    def send(event=None):
        text = entry.get().strip()
        if not text:
            return
        add("You", text)
        add("ShopBot", bot.get_response(text))
        entry.delete(0, tk.END)

    entry.bind("<Return>", send)
    tk.Button(root, text="Send", command=send).pack(side=tk.LEFT, pady=(0, 10))
    add("ShopBot", "Hi! Ask me about orders, shipping, returns or payments.")
    root.mainloop()


if __name__ == "__main__":
    bot = FAQChatbot(Path(__file__).parent / "faqs.json")
    run_gui(bot) if "--gui" in sys.argv else run_cli(bot)
