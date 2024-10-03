from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


class SentimentAnalyzer:
    def __init__(self):
        self.vectorizer = CountVectorizer(stop_words='english')
        self.classifier = MultinomialNB()
        self.is_trained = False

    def analyze_sentiment(self, text):
        # Use TextBlob for quick sentiment analysis
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity

        if polarity > 0:
            return 'positive'
        elif polarity < 0:
            return 'negative'
        else:
            return 'neutral'

    def train_classifier(self, texts, labels):
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

        # Vectorize text data
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        X_test_vectorized = self.vectorizer.transform(X_test)

        # Train the model
        self.classifier.fit(X_train_vectorized, y_train)

        # Evaluate the model
        y_pred = self.classifier.predict(X_test_vectorized)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        print(f"Model Accuracy: {accuracy}")
        print("Classification Report:")
        print(report)

        self.is_trained = True

    def predict_sentiment(self, text):
        if not self.is_trained:
            raise ValueError("Classifier has not been trained yet. Please train the model first.")

        vectorized_text = self.vectorizer.transform([text])
        prediction = self.classifier.predict(vectorized_text)
        return prediction[0]


if __name__ == "__main__":
    analyzer = SentimentAnalyzer()

    # Test TextBlob-based sentiment analysis
    sample_text = "I love this product! It's amazing."
    print(f"TextBlob Sentiment: {analyzer.analyze_sentiment(sample_text)}")

    # Example of training and using the classifier
    # Note: You would need a labeled dataset for actual training
    texts = ["I love this", "I hate this", "This is okay", "Great product", "Terrible experience"]
    labels = ["positive", "negative", "neutral", "positive", "negative"]

    analyzer.train_classifier(texts, labels)
    print(f"Classifier Sentiment: {analyzer.predict_sentiment('This is a fantastic product!')}")