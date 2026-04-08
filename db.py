import os
from motor.motor_asyncio import AsyncIOMotorClient

MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB  = os.getenv("MONGODB_DB", "workping")

_client: AsyncIOMotorClient | None = None


def get_embeddings_collection():
    global _client
    if _client is None:
        if not MONGODB_URI:
            raise RuntimeError("MONGODB_URI environment variable is not set")
        _client = AsyncIOMotorClient(MONGODB_URI)
    return _client[MONGODB_DB]["faceembeddings"]
