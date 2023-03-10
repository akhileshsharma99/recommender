import os
from recommender import Recommender


def main():
    api_key = os.environ['OPENAI_API_KEY']
    datasets_directory = 'data/amazon_reviews_text'
    db_directory = 'db/amazon_reviews'

    query = "pizza not whole wheat"

    gpt = Recommender(api_key=api_key)
    embedding = gpt.createEmbedding()

    if os.path.exists(db_directory):
        print('database exists...')
        db = gpt.getDatabase(embedding=embedding, db_dir=db_directory)
    else:
        print('database does not exists...')
        docs = gpt.createDocs(datasets_directory=datasets_directory)
        db = gpt.createDatabase(docs=docs, embedding=embedding, db_dir=db_directory)
        
    results = gpt.query_database(query, db)

    for result in results:
        print(result.page_content)
        print()

def add_documents():
    api_key = os.environ['OPENAI_API_KEY']
    datasets_directory = 'data/amazon_reviews_test_text'
    db_directory = 'db/amazon_reviews_test'

    query = "HP laptop with good specs for programming"

    gpt = Recommender(api_key=api_key)
    embedding = gpt.createEmbedding()

    if os.path.exists(db_directory):
        print('database exists...')
        docs = gpt.createDocs(datasets_directory=datasets_directory)
        db = gpt.getDatabase(embedding=embedding, db_dir=db_directory)
        db.add_texts(texts=docs)
    else:
        print('database does not exists...')
        return
        
    results = gpt.query_database(query, db)

    print(results[0].page_content)
    print(results)

if __name__ == "__main__":
    main()
    # add_documents()
