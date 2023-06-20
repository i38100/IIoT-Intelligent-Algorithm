from Dataset import DataType
from utils import unique_id
from ServerBase import ServerBase
from resource_pool import resources

class CommonServer(ServerBase):
    """抽象基类，完成一些公共操作
    """
    type_ = "common"
    def __init__(self, cluster) -> None:
        super().__init__(cluster)
        self.id = resources.get_id('server', self)

class EdgeServer(CommonServer):
    type_ = "edge"
    def __init__(self, cluster):
        super().__init__(cluster)
        
class CloudServer(CommonServer):
    type_ = "cloud"
    def __init__(self, cluster):
        super().__init__(cluster)
        
class EndDevice(CommonServer):
    type_ = "end"
    mips: int = 1              # 假设端设备的计算能力较低
    capacity: int = 0          # 数据容量，这里假设端设备不储存数据
    def __init__(self, cluster):
        super().__init__(cluster)