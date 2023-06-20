class ServerBase:
    """表示服务器的抽象基类
    """
    type_ = "base"
    def __init__(self, cluster):
        self.mips: int = 10                  # 每秒可执行x百万条指令
        self.bandwidth: float = 10           # 外网通信的带宽
        self.memory: float = 4096            # MB
        self.capacity: int = 200             # 数据容量
        self.capacity_usage = 0              # 已经使用的容量
        self.security: float = 90            # 服务器可信度, 最高100
        self.cost: float = 0.1               # 服务器每小时价格
        self.max_power: int = 65             # 满载时功耗
        self.base_power: int = 5             # 待机功耗        
        self.cluster = cluster               # 所属集群
    
    def __repr__(self) -> str:
        return """Server type: {}, id: {}, capacity: {}, usage: {}""".format(self.type_, self.id, self.capacity, self.capacity_usage)
    
    def __str__(self) -> str:
        return self.__repr__()
    
    def powerOn(self):
        pass
    
    def powerOff(self):
        pass
    
    def fetch(self, id: int, size: int, data_type):
        """
        从编号为id的主机中获取大小为size的数据，类型为data_type
        """
        pass
    
    def execute(self, task):
        pass