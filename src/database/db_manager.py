import psycopg2
import yaml

class DBManager:
    def __init__(self, config_path):
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
        self.conn = psycopg2.connect(
            dbname=config['database']['dbname'],
            user=config['database']['user'],
            password=config['database']['password'],
            host=config['database']['host'],
            port=config['database']['port']
        )
        self.cursor = self.conn.cursor()
        # Create table if it does not exist
        self.create_table()

    def create_table(self):
        create_table_query = """
        CREATE TABLE IF NOT EXISTS processed (
            id SERIAL PRIMARY KEY,
            text TEXT UNIQUE,
            cleaned_text TEXT,
            sentiment VARCHAR(50)
        );
        """
        self.cursor.execute(create_table_query)
        self.conn.commit()

    def save_post(self, post_data, cleaned_text, sentiment):
        insert_query = """
        INSERT INTO processed (text, cleaned_text, sentiment)
        VALUES (%s, %s, %s)
        ON CONFLICT (text)
        DO UPDATE SET
            cleaned_text = EXCLUDED.cleaned_text,
            sentiment = EXCLUDED.sentiment
        """
        self.cursor.execute(insert_query, (
            post_data['text'], cleaned_text, sentiment
        ))
        self.conn.commit()

    def get_all_posts(self):
        self.cursor.execute("SELECT * FROM processed")
        posts = self.cursor.fetchall()
        return posts

    def get_sentiment_distribution(self):
        self.cursor.execute("""SELECT sentiment, COUNT(id) FROM processed GROUP BY sentiment""")
        distribution = self.cursor.fetchall()
        return dict(distribution)

    def close(self):
        self.cursor.close()
        self.conn.close()