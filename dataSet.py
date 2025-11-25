from abc import ABC, abstractmethod

class DataSet(ABC):
    def __init__(self, dataset_name, split, max_samples=None, dataset_dir=None):
        self.dataset_name = dataset_name
        self.split = split
        self.max_samples = max_samples
        self.dataset_dir = dataset_dir
    @abstractmethod
    def load_dataset(self):
        pass
    @abstractmethod
    def prepare_annotations(self, overwrite=False):
        pass