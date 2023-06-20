from typing import Set, List
from ServerBase import ServerBase


class TaskBase:
    """表示Task的基类
    """
    def __init__(self, name, ins=100) -> None:
        self.m_instructions = ins
        self.end_time: float =   -1
        self.start_time: float = -1
        self.name = name
        self.done = False
        self.m_instructions: int = 30                               # 程序运行所需平均指令数
        self.server: ServerBase = None                              # 运行位置
        self.data_transmission_time: float = 0                      # 数据传输所需时间

    def __repr__(self) -> str:
        return """Task name: {}, id: {}, dependencies: {} datasets and {} tasks, done: {}""".format(self.name, self.id, 
                                len(self.data_dependencies), len(self.task_dependencies), self.done)
    def __str__(self) -> str:
        return self.__repr__()
    
    def depend_task(self, *args):
        pass

    def depend_data(self, *args):
        pass
            
    def start(self, start_time):
        pass
    
    def request_data(self):
        pass
    
    def compute():
        pass
    
    def run(self):
        pass