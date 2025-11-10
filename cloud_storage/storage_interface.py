import abc

class CloudStorageInterface(abc.ABC):
    @abc.abstractmethod
    def upload(self, file_path, file_name):
        pass
    @abc.abstractmethod
    def download(self, file_name, file_path):
        pass
    @abc.abstractmethod
    def delete(self, file_name):
        pass
    @abc.abstractmethod
    def list_files(self):
        pass
    @abc.abstractmethod
    async def bulk_get(self, keys: list[str]) -> list[bytes]:
        pass
    @abc.abstractmethod
    def bulk_get_sync(self, keys: list[str]) -> list[bytes]:
        pass