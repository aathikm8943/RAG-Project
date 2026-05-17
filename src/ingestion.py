import re
import os
from dataclasses_json import config
import yaml
import numpy as np

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import ollama

from milvus_setup import MilvusSetup

class PDFIngestion:
    def __init__(self):
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, 'config.yml')
        
        print(f"Loading configuration from: {config_path}")
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def _clean_text(self, text):
        text = text.replace('\xa0', ' ')  # Remove non-breaking spaces (\xa0)
        text = re.sub(r'[ \t]+', ' ', text) # Remove multiple spaces/tabs but preserve newlines
        text = re.sub(r'\n+', '\n', text) # Remove excessive newlines
        text = re.sub(r'Page \d+', '', text) # Remove page numbers
        text = re.sub(r'http\S+|www\S+', '', text) # Remove URLs

        return text.strip()
    
    def read_pdfs(self):    
        for filename in os.listdir(self.config['FOLDER_PATH']):
            if filename.endswith('.pdf'):
                file_path = os.path.join(self.config['FOLDER_PATH'], filename)
                loader = PyMuPDFLoader(file_path=file_path)
                print(f"Loading {filename}...")
                docs = loader.load()
                cleaned_docs = [self._clean_text(doc.page_content) for doc in docs] 
                print(f"Loaded {len(cleaned_docs)} documents from {filename}")

        return cleaned_docs
    
    def fixed_size_chunking(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=self.config['CHUNK_SIZE'], chunk_overlap=self.config['CHUNK_OVERLAP'])
        combined_text = "\n\n".join(self.read_pdfs())
        chunks = splitter.split_text(combined_text)
        
        return chunks
    
    def convert_embeddings(self, text):
        response = ollama.embeddings(model=self.config['EMBEDDING_MODEL'], prompt=text)
        return np.array(response['embedding'])
    
    def convert_embeddings_for_milvus(self, chunks):
        conv_embed = []
        for chunk in chunks:
            embeddings = self.convert_embeddings(text=chunk)
            conv_embed.append(embeddings)
        return conv_embed
    
    def combined_text_chunks(self):
        combined_text = "\n\n".join(self.read_pdfs())
        sent_embeddings = [self.convert_embeddings(sent) for sent in sent_tokenize(combined_text)]
        
        return combined_text, sent_embeddings
    
    def semantic_chunking(self):
        combined_text, sent_embeddings = self.combined_text_chunks()
        sentences = sent_tokenize(combined_text)

        similarity_chunks = []
        current_chunk = [sentences[0]]

        for i in range(1, len(sentences)):

            similarity = cosine_similarity(
                [sent_embeddings[i - 1]],
                [sent_embeddings[i]]
            )[0][0]
            
            # Topic changed
            if similarity < self.config['THRESHOLD']:

                similarity_chunks.append(" ".join(current_chunk))
                current_chunk = [sentences[i]]

            else:
                current_chunk.append(sentences[i])

        # Add final chunk
        similarity_chunks.append(" ".join(current_chunk))
        
        return similarity_chunks
    
    def convert_embeddings_for_milvus(self, chunks):
        conv_embed = []
        for chunk in chunks:
            embeddings = self.convert_embeddings(prompt=chunk)
            conv_embed.append(embeddings)
        return conv_embed
    
    def insert_data(self, collection, chunks=None, chunk_type=None):
        if not chunks:
            print("Error: No chunks provided!")
            return 0, "No data to insert"
        
        texts = []
        chunk_types = []
        embeddings = []

        for i, chunk in enumerate(chunks):
            try:
                converted_embed = self.convert_embeddings(text=chunk)
                texts.append(chunk)
                chunk_types.append(chunk_type)
                embeddings.append(converted_embed)

            except Exception as e:
                print(f"Error processing chunk {i}: {e}")

        print(f"Preparing to insert {len(texts)} records...")
        
        try:
            result = collection.insert([texts, chunk_types, embeddings])
            print(f"Insert result: {result}")
            
            # Flush to ensure data is persisted
            collection.flush()
            print(f"Entities after flush: {collection.num_entities}")
            
            return collection.num_entities, "Successfully inserted data into Milvus collection"
        
        except Exception as e:
            print(f"Insert failed: {e}")
            return 0, f"Insert failed: {str(e)}"
        
    def ingest_to_milvus(self, collection):
        ## similarity_chunks ingestion
        similarity_chunks = self.semantic_chunking()
        num_entities, message = self.insert_data(collection, chunks=similarity_chunks, chunk_type="semantic")
        print(f"\nIngestion result: {num_entities} entities - {message}")
        
        ## fixed-size chunks ingestion
        fixed_chunks = self.fixed_size_chunking()
        num_entities, message = self.insert_data(collection, chunks=fixed_chunks, chunk_type="recursive_character")
        print(f"\nIngestion result: {num_entities} entities - {message}")
        
if __name__ == "__main__":
    milvus_setup = MilvusSetup()
    milvus_setup.connect()
    collection = milvus_setup.collection_creation()

    ingestion = PDFIngestion()
    ingestion.ingest_to_milvus(collection)

    milvus_setup.disconnect()
    