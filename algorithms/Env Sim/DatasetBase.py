class DatasetBase:    
    def __init__(self, name, size, privacy=False) -> None:
        self.name = name
        self.size = size
        self.privacy: bool = privacy       # 是否为隐私数据
