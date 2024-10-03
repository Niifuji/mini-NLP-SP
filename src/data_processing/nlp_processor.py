import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re
from collections import Counter


class NLPProcessor:
    def __init__(self):
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.classifier = MultinomialNB()

    # Preprocess text by converting to lowercase, removing special characters, tokenizing, removing stopwords, and lemmatizing
    def preprocess_text(self, text):
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords and lemmatize
        cleaned_tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        return ' '.join(cleaned_tokens)

    # Analyze sentiment of text using TextBlob
    def analyze_sentiment(self, text):
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        if polarity > 0.1:
            return 'positive'
        elif polarity < -0.1:
            return 'negative'
        else:
            return 'neutral'

    # Extract features from text using TF-IDF vectorization
    def extract_features(self, texts):
        return self.vectorizer.fit_transform(texts)

    # Train a classifier on the given texts and labels
    def train_classifier(self, texts, labels):
        X = self.extract_features(texts)
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
        # Train the classifier
        self.classifier.fit(X_train, y_train)
        # Evaluate the classifier
        y_pred = self.classifier.predict(X_test)
        print(classification_report(y_test, y_pred))

    # Predict sentiment of a given text
    def predict_sentiment(self, text):
        preprocessed_text = self.preprocess_text(text)
        features = self.vectorizer.transform([preprocessed_text])
        return self.classifier.predict(features)[0]

    # Get the top N words from a list of texts
    def get_top_n_words(self, texts, n=10):
        all_words = []
        for text in texts:
            all_words.extend(self.preprocess_text(text).split())
        return Counter(all_words).most_common(n)


if __name__ == "__main__":
    processor = NLPProcessor()
    # Test preprocessing
    sample_text = "Very bad"
    print(f"Preprocessed text: {processor.preprocess_text(sample_text)}")
    # Test sentiment analysis
    print(f"Sentiment: {processor.analyze_sentiment(sample_text)}")
    # Test classifier (you would need a labeled dataset for this)
    texts = ["I love this product", "This is terrible", "Not bad, could be better", "Amazing experience!",
             "Worst purchase ever"]
    labels = ["positive", "negative", "neutral", "positive", "negative"]
    processor.train_classifier(texts, labels)
    # Test prediction
    new_text = "This product exceeded my expectations"
    print(f"Predicted sentiment: {processor.predict_sentiment(new_text)}")
    # Test top words
    print(f"Top words: {processor.get_top_n_words(texts)}")


