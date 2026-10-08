# CodeAlpha_FAQChatbot

FAQ chatbot for an online store, built for the **CodeAlpha Artificial Intelligence Internship – Task 2: Chatbot for FAQs**.

**Author:** Ayan Kumar

## How it works
FAQs (`faqs.json`) → NLTK preprocessing (tokenize, remove stopwords, lemmatize) → TF-IDF vectors → cosine similarity → best-matching answer.

## Run
```
pip install -r requirements.txt
python chatbot.py          # terminal chat
python chatbot.py --gui    # Tkinter chat window
```
Edit `faqs.json` to use your own topic.

## Files
- `chatbot.py` – chatbot code (preprocessing, matching, terminal and chat-window UI)
- `faqs.json` – the questions and answers
- `requirements.txt` – required libraries
