import csv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import DeepLake
from langchain.vectorstores import VectorStore
from langchain.document_loaders import UnstructuredFileLoader
import os

from exceptions import PathNotDirectoryException


class Recommender:

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def query_database(self, query: str, db: VectorStore | DeepLake, k: int = 4):
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