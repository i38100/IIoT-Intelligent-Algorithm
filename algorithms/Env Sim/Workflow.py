from Dataset import DataType
from Task import Task
from config import data_type_num, task_type_num
from typing import List, Set



class Trace:
    pass
    
class TraceExample(Trace):
    def __init__(self) -> None:
        self.trace: List[Task] = []
        self.task_done: Set[Task] = set()
        self.data_s = {}
        self.task_s = {}
        for i in range(data_type_num):
            self.data_s['d'+str(i)] = DataType('d'+str(i), 50+i*20)
        for i in range(task_type_num):
            self.task_s['t'+str(i)] = Task('t'+str(i), 100+i*20)
        self._build_example()
        pass
    
    def _build_example(self):
        self.task_s['t1'].depend_data(self.data_s['d0'])
        self.task_s['t2'].depend_data(self.data_s['d1'], self.data_s['d2'])
        self.task_s['t3'].depend_data(self.data_s['d3'])
        self.task_s['t2'].depend_task(self.task_s['t1'])
        self.task_s['t3'].depend_task(self.task_s['t2'])
        
        task1 = Task('A')
        task2 = Task('B')
        task3 = Task('C')
        
        self.trace = [task1, task2, task3]
        
    def evict(self) -> Task:
        """
        返回需要执行的task
        """
        for task in self.trace:
            if task.done:
                continue
            dependencies: Set[Task] = task.task_dependencies
            have_done: Set[Task] = dependencies & self.task_done
            if len(have_done) == len(dependencies):
                # 所以依赖任务都完成了，可以运行
                end_time = -1
                for ele in dependencies:
                    end_time = end_time if end_time > ele.end_time else ele.end_time
                task.start_time = end_time        # 取最后的依赖任务完成时间为新任务的开始时间
                return task                       # 一次只取一个
            

trace_example = TraceExample()          # 单例
