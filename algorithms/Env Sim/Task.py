from utils import unique_id
from resource_pool import resources
from typing import Set, List
from Dataset import DataType, Dataset
from Server import ServerBase
from TaskBase import TaskBase


class CommonTask(TaskBase):
    def __init__(self, name, ins=100) -> None:
        super().__init__(name, ins)
        self.data_dependencies: Set["Dataset"] = set()              # 依赖的数据集实例
        self.task_dependencies: Set["TaskBase"]    = set()          # 依赖的task实例
        self.id: int = resources.get_id('task', self)

    def depend_task(self, *args):
        for task in args:
            if type(task) == Task:
                self.task_dependencies.add(task)

    def depend_data(self, *args):
        for data in args:
            if type(data) == DataType:
                self.data_dependencies.add(Dataset(data))
            
    def start(self, start_time):
        self.start_time = start_time
    
    def request_data(self):
        # for data in self.data_dependencies:
            # self.data_transmission_time += 
        pass
    
    def compute():
        pass
    
    def run(self):
        self.request_data()
        self.compute()
        self.done = True
    
    
class Task(CommonTask):
    def __init__(self, name, ins=100) -> None:
        super().__init__(name, ins)