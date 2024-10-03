import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
class TextCleaner:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    def clean_text(self, text):
        #Convert to lowercase
        text = text.lower()
        #Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        #Remove stopwords and lemmatize
        cleaned_tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        return ' '.join(cleaned_tokens)
if __name__ == "__main__":
    cleaner = TextCleaner()
    sample_text = "This is a sample text with some numbers 123 and special characters !@#."
    cleaned_text = cleaner.clean_text(sample_text)
    print(f"Original: {sample_text}")
    print(f"Cleaned: {cleaned_text}")