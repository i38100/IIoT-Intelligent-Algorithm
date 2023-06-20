# 负责全局的资源注册和管理
from typing import Dict, List
from ServerBase import ServerBase
from TaskBase   import TaskBase
from DatasetBase import DatasetBase

class ResourcePool:
    server_pool: Dict[int, ServerBase] =  {}
    data_pool: Dict[int, DatasetBase]   =  {}
    task_pool: Dict[int, TaskBase]   =  {}
    
    server_id: int = -1
    data_id: int   = -1
    task_id: int   = -1
    common_id: int = -1
    
    def __init__(self) -> None:
        pass
    
    def get_id(self, type_, resource):
        if type_ == 'server':
            self.server_id += 1
            self.server_pool[self.server_id] = resource
            return self.server_id
        elif type_ == 'data':
            self.data_id += 1
            self.data_pool[self.data_id] = resource
            return self.data_id
        elif type_ == 'task':
            self.task_id += 1
            self.task_pool[self.task_id] = resource
            return self.task_id
        else:
            self.common_id += 1
            return self.common_id
        
    def get_servers(self, type_="server"):
        all_ = list(self.server_pool.keys())
        ends = [x for x in all_ if self.server_pool[x].type_ == 'end']
        edges = [x for x in all_ if self.server_pool[x].type_ == 'edge']
        clouds = [x for x in all_ if self.server_pool[x].type_ == 'cloud']
        if type_ == 'cloud':
            return clouds
        elif type_ == 'edge':
            return edges
        elif type_ == 'end':
            return ends
        elif type_ == 'server':
            res = []
            res.extend(clouds)
            res.extend(edges)
            return res
        elif type_ == 'all': 
            res = []
            res.extend(ends)
            res.extend(edges)
            res.extend(clouds)
            return res
        
        
resources = ResourcePool()   # 一个单例模式的对象

