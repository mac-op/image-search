import os

from storage_interface import CloudStorageInterface

def get_storage_client() -> CloudStorageInterface:
    if os.getenv("USE_GCP") or True:
        import gcloud
        return gcloud.GCloudStorage()
    raise RuntimeError("No cloud storage provider specified")
