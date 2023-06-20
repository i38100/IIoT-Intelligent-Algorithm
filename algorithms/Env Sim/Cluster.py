from Server import *
from utils import unique_id
from resource_pool import resources

class Cluster:
    type_ = "base"
    def __init__(self, size):
        self.id = resources.get_id('cluster', self)
        self.size = size
        self.size: int = 10                      # 区域内含有的服务器数量
        self.bandwidth_inner: float = 100        # 内部通信带宽
        self.delay_inner: float = 0.1            # 内部时延
    
    def __repr__(self) -> str:
        return """Server Cluster type: {}, id: {}, size: {}""".format(self.type_, self.id, self.size)
    
    def __str__(self) -> str:
        return self.__repr__()

class EdgeRegion(Cluster):
    type_ = "edge"
    def __init__(self, size):
        super().__init__(size)
        self.bandwidth_inner = 100
        self.delay_inner = 0.1
        self.edge_servers = []                     # 区域内的边缘节点
        self.end_devices = []                      # 区域内的端设备
        for i in range(size):
            self.edge_servers.append(EdgeServer(cluster=self))
        
    def build_end_device(self, size):
        for i in range(size):
            self.end_devices.append(EndDevice(cluster=self))
    
class CloudDataCenter(Cluster):
    type_ = "cloud"
    def __init__(self, size):
        super().__init__(size)
        self.bandwidth_inner = 200
        self.delay_inner = 0.1
        self.cloud_servers = []
        for i in range(size):
            self.cloud_servers.append(CloudServer(cluster=self))
            