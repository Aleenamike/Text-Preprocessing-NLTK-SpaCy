# Text Preprocessing using NLTK and spaCy

This project demonstrates basic text preprocessing techniques on a small text corpus using **NLTK** and **spaCy** in Python.

## 📌 Steps Implemented
1. **Tokenization**  
   - NLTK: `word_tokenize()`  
   - spaCy: `nlp(text)`  

2. **Stemming** (NLTK only)  
   - Porter Stemmer  
   - Snowball Stemmer  

3. **Lemmatization**  
   - NLTK: `WordNetLemmatizer`  
   - spaCy: `token.lemma_`  

4. **Stopword Removal**  
   - NLTK: `stopwords.words('english')`  
   - spaCy: `token.is_stop`

## ⚙️ Requirements
- Python 3.x  
- Libraries: `nltk`, `spacy`

Install dependencies:
```bash
pip install nltk spacy
python -m spacy download en_core_web_sm

Running the Script
python text_preprocessing.py

Example Corpus
News and social media related sample sentences are used.

Output
Shows tokenization, stemming, lemmatization, and stopword removal results for both NLTK and spaCy.
