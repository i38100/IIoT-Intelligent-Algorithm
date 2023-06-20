from DatasetBase import DatasetBase
from resource_pool import resources
from utils import unique_id


class DataType(DatasetBase):
    """数据集类型
    """
    def __init__(self, name, size, privacy=False) -> None:
        super().__init__(name, size, privacy)
        self.id = resources.get_id('data', self)
        
    def __repr__(self) -> str:
        return """A Dataset type named {} and size of {}""".format(self.name, self.size)
        
class Dataset:
    """数据集实例，聚合模式
    """
    def __init__(self, data_type: DataType, name=None, size=None) -> None:
        self.id = resources.get_id('dataset', self)
        self.type_: DataType = data_type
        self.size = size if size else data_type.size
        self.name = name if name else data_type.name + ' instance'

    def __repr__(self) -> str:
        return """A Dataset instance of type {} and size of {}""".format(self.type_.name, self.type_.size)