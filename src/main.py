import yaml
from data_processing.nlp_processor import NLPProcessor
# from database.db_manager import DBManager
import json
import os
import pandas as pd


def main():
    # Load configuration from YAML file
    with open('config/config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)
    json_file_path = "./data/raw/formula1_posts.json"  # Currently using a sample file
    if not os.path.exists(json_file_path):
        print(f"File {json_file_path} does not exist. Please scrape data...")
        return

    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        posts = json.load(json_file)
    print(f"Loaded {len(posts)} posts from {json_file_path}.")

    nlp_processor = NLPProcessor()
    processed_posts = []
    for post in posts:
        post_text = post['text']
        cleaned_text = nlp_processor.preprocess_text(post_text)
        sentiment = nlp_processor.analyze_sentiment(post_text)
        post['cleaned_text'] = cleaned_text
        post['sentiment'] = sentiment
        processed_posts.append(post)

    # Save data to the database
    # uncomment the following lines to save data to the postgresql database
    # db_manager = DBManager('config/config.yaml')
    # for post in processed_posts:
    #     db_manager.save_post(post, cleaned_text=post['cleaned_text'], sentiment=post['sentiment'])
    # db_manager.close()

    # Create DataFrame from processed_posts
    df = pd.DataFrame(processed_posts, columns=['text', 'cleaned_text', 'sentiment'])

    csv_file_path = './data/processed/processed.csv'
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
    df.to_csv(csv_file_path, index=False)
    print(f"Data saved to {csv_file_path}")


if __name__ == "__main__":
    main()