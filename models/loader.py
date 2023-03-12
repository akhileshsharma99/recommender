import os
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter

from exceptions import PathNotDirectoryException


class Loader:
    def __init__(self, input_dir: str) -> None:
        self._input_dir = input_dir

    def createDocs(self):
        def getFilesInDirectory(directory_path: str) -> list[str]:
            return [os.path.join(directory_path, filename) for filename in os.listdir(directory_path)]

        if os.path.isdir(self._input_dir):

            docs = []
            file_paths = getFilesInDirectory(self._input_dir)

            for file_path in file_paths:
                loader = UnstructuredFileLoader(file_path, mode="elements")
                documents = loader.load()

                # Split the text into smaller chunks for better performance
                text_splitter = CharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=0)
                docs.extend(text_splitter.split_documents(documents))
            return docs
        raise PathNotDirectoryException(path=self._input_dir)
