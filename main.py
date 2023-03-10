import os
from llm import LLM


def main():
    api_key = os.environ['OPENAI_API_KEY']
    datasets_directory = 'data/amazon_reviews_test_text'
    db_directory = 'db/amazon_review'

    query = "Dog food"

    gpt = LLM(api_key=api_key)
    embedding = gpt.createEmbedding()

    if os.path.exists(db_directory):
        db = gpt.getDatabase(embedding=embedding, db_dir=db_directory)
    else:
        docs = gpt.createDocs(datasets_directory=datasets_directory)
        db = gpt.getDatabase(docs=docs, embedding=embedding, db_dir=db_directory)
        
    results = gpt.query_database(query, db)

    print(results[0].page_content)
    print(results)


if __name__ == "__main__":
    main()
