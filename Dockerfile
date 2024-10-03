
FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    && pip install nltk

RUN python -m nltk.downloader punkt punkt_tab stopwords wordnet

COPY . .

RUN pip install -r requirements.txt

CMD ["python", "src/main.py"]
