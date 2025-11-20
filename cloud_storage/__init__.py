import os

from .storage_interface import CloudStorageInterface

def get_storage_client() -> CloudStorageInterface:
    if os.getenv("USE_GCP") or True:
        from .gcloud_storage import GCloudStorage
        return GCloudStorage()
    raise RuntimeError("No cloud storage provider specified")
