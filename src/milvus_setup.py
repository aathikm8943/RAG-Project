from pymilvus import connections
from pymilvus import (
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection
)

class MilvusSetup:
    def __init__(self, host="localhost", port="19530"):
        self.host = host
        self.port = port

    def connect(self):
        connections.connect(
            alias="default",
            host=self.host,
            port=self.port
        )

    def disconnect(self):
        connections.disconnect(alias="default")
        
    def collection_creation(self):

        fields = [

            FieldSchema(
                name="id",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True
            ),

            FieldSchema(
                name="text",
                dtype=DataType.VARCHAR,
                max_length=5000
            ),

            FieldSchema(
                name="chunk_type",
                dtype=DataType.VARCHAR,
                max_length=100
            ),

            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=768
            )
        ]

        schema = CollectionSchema(fields)

        collection = Collection(
            name="nutrition_rag",
            schema=schema
        )
        
        return collection