import os

if __name__ == "__main__":
    if os.getenv("USE_GCP") or True:
        from service import app
