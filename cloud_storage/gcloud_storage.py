import os
from typing import Any
import asyncio

from google.cloud.storage import Blob

from .storage_interface import CloudStorageInterface
from google.cloud import storage

class GCloudStorage(CloudStorageInterface):
    BUCKET_NAME = os.environ.get("BUCKET_NAME")

    def __init__(self):
        self.client = storage.Client()
        self.bucket = self.client.bucket(self.BUCKET_NAME)

    def upload(self, file_path, file_name):
        Blob(self.bucket, file_name).upload_from_filename(file_name)

    def download(self, file_name, _) -> bytes:
        return Blob(self.bucket, file_name).download_as_bytes()

    def delete(self, file_name):
        Blob(self.bucket, file_name).delete()

    def list_files(self):
        pass

    async def bulk_get(self, keys: list[str]) -> list[bytes]:
        async def fetch_one(key: str) -> bytes:
            loop = asyncio.get_event_loop()
            blob = Blob(self.bucket, key)
            data = await loop.run_in_executor(None, blob.download_as_bytes)
            return data

        results = await asyncio.gather(*[fetch_one(key) for key in keys])
        return results

    def bulk_get_sync(self, keys: list[str]) -> list[bytes]:
        return [Blob(self.bucket, key).download_as_bytes() for key in keys]
