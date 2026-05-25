import warnings
from pymilvus import MilvusClient
from pymilvus.orm import utility

# Suppress PyMilvus deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pymilvus")

class MilvusSetup:
    def __init__(self, host="localhost", port="19530"):
        self.host = host
        self.port = port
        self.client = MilvusClient(uri=f"http://{host}:{port}", token="")

    def create_collection(self, collection_name: str, index_params: dict):
        """Create collection with specified index parameters."""
        try:
            # Check if collection exists
            if self.client.has_collection(collection_name):
                print(f"Collection '{collection_name}' already exists, dropping...")
                self.client.drop_collection(collection_name)
            
            # Create collection with schema
            self.client.create_collection(
                collection_name=collection_name,
                dimension=768,
                metric_type="COSINE",
                auto_id=True,
                vector_field_name="embedding",
                primary_field_name="id",
                id_type="int64"
            )
            
            # Create index
            self.client.create_index(
                collection_name=collection_name,
                field_name="embedding",
                index_params=index_params
            )
            
            print(f"Collection '{collection_name}' created successfully")
            return True
            
        except Exception as e:
            print(f"Error creating collection: {e}")
            raise