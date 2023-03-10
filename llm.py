import csv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import DeepLake
from langchain.vectorstores import VectorStore
from langchain.document_loaders import UnstructuredFileLoader
import os

from exceptions import PathNotDirectoryException


class LLM:

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def query_database(self, query: str, db: VectorStore, k: int = 4):
        return db.similarity_search(query, k=k)

    def getDatabase(self, embedding: OpenAIEmbeddings, db_dir: str):
        db = DeepLake(dataset_path=db_dir, embedding_function=embedding)
        return db
    
    def createDatabase(self, docs: list, embedding: OpenAIEmbeddings, db_dir: str):
        db = DeepLake.from_documents(
                documents=docs, embedding=embedding, dataset_path=db_dir)
        return db
    
    def createEmbedding(self):
        # Create an instance of OpenAIEmbeddings using the API
        embeddings = OpenAIEmbeddings(openai_api_key=self._api_key)
        return embeddings

    @staticmethod
    def createDocs(datasets_directory: str):
        def getFilesInDirectory(directory_path: str) -> list[str]:
            return [os.path.join(directory_path, filename) for filename in os.listdir(directory_path)]

        if os.path.isdir(datasets_directory):

            docs = []
            file_paths = getFilesInDirectory(datasets_directory)

            for file_path in file_paths:
                loader = UnstructuredFileLoader(file_path, mode="elements")
                documents = loader.load()

                # Split the text into smaller chunks for better performance
                text_splitter = CharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=0)
                docs.extend(text_splitter.split_documents(documents))
            return docs
        raise PathNotDirectoryException(path=datasets_directory)


# datasets_directory = 'nutrition/datasets/kaggle_food_nutrition'
# persist_directory = 'nutrition/db/kaggle_food_nutrition'


# # If the vector database has already been created, load it from disk
# if os.path.exists(persist_directory):
#     vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
# else:
#     # If the vector database hasn't been created, load the dataset from disk
#     loader = DirectoryLoader(datasets_directory, glob='**/*.txt')
#     documents = loader.load()

#     # Split the text into smaller chunks for better performance
#     text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
#     texts = text_splitter.split_documents(documents)

#     # Create the vector database from the documents
#     vectordb = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=persist_directory)

# # Create a VectorDBQA instance for answering questions
# qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=vectordb , return_source_documents=True)

# # Ask a question and print the result and source documents
# query = "200g of lean beef how many calories"
# result = qa({"query": query})
# print(result['result'])
# print(result['source_documents'])
