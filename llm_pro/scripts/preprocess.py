# Class 기반 분할

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class SajuPreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path #

    def process(self):
        loader = TextLoader(self.file_path, encoding="utf-8") #
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50) #
        return text_splitter.split_documents(documents)