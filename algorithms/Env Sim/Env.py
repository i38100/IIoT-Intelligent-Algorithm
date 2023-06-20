import logging
import random
from Cluster import CloudDataCenter, EdgeRegion, Cluster
from Server import ServerBase
from utils import random_split, random_gauss
from Task import Task
from Dataset import DataType
from config import *
from typing import List, Dict, Tuple
from resource_pool import resources

class TypeUnreasonableException(Exception):
    def __init__(self, src: ServerBase, des: ServerBase):
        self.src = src
        self.des = des
    
    def __str__(self):
        print("不应该出现的节点类型组合")
        print(self.src, self.des)

class Env:
    def __init__(self):
        """云-边-端环境
        Args:
            mode (_type_): 网络生成模式
        """
        self.edge_layer: List[EdgeRegion] = []                    # EdgeRegion 的列表
        self.data_map = {}                                        # 各类数据存放的列表
        self.task_map = {}
        self.cloud_data_center: CloudDataCenter = None            # 云数据中心        
        self._build_env(3, 20, 3)
    
    def __repr__(self) -> str:
        return "The Cloud-Edge-End Env is constructed."
    
    def __str__(self) -> str:
        return self.__repr__()
    
    def sim_random_trans(self, src: ServerBase, des: ServerBase):
        """随机产生链路延迟与带宽
        Args:
            src (ServerBase): 源主机
            des (ServerBase): 目标主机

        Returns:
            _type_:二元组 (延迟，带宽)
        """
        src_region: Cluster = src.cluster
        des_region: Cluster = des.cluster
        try:
            if src_region.id == des_region.id:
                delay, bw = src_region.delay_inner, src_region.bandwidth_inner
            
            elif src_region.type_ == des_region.type_:
                # 都在边缘区
                delay, bw = random_gauss(delay_edge_to_edge), random_gauss(bw_edge_to_edge)
            
            elif src.type_ == 'end':
                if des.type_ == 'edge':
                    delay, bw = random_gauss(delay_edge_to_end), random_gauss(bw_edge_to_end)
                elif des.type == 'cloud':
                    delay, bw = random_gauss(delay_cloud_to_end), random_gauss(bw_cloud_to_end)
                else:
                    raise TypeUnreasonableException(src, des)
            elif src.type_ == 'edge':
                if des.type_ == 'edge':
                    delay, bw = random_gauss(delay_edge_to_edge), random_gauss(bw_edge_to_edge)
                elif des.type == 'cloud':
                    delay, bw = random_gauss(delay_cloud_to_edge), random_gauss(bw_cloud_to_edge)
                else:
                    raise TypeUnreasonableException(src, des)                
            elif src.type_ == 'cloud':
                if des.type_ == 'edge':
                    delay, bw = random_gauss(delay_cloud_to_edge), random_gauss(bw_cloud_to_edge)
                else:
                    raise TypeUnreasonableException(src, des)
            else:
                raise TypeUnreasonableException(src, des)
        except TypeUnreasonableException:
            delay, bw = random_gauss(delay_common), random_gauss(bw_common)

        logging.debug("%s <-> %s, delay: %f, bandwidth: %f", src.type_, des.type_, delay, bw)
        return delay, bw

    def _build_env(self, edge_region_num=EDGE_REGION_NUM, edge_server_num=EDGE_SERVER_NUN, cloud_server_num=CLOUD_SERVER_NUM):
        """随机生成一个云-边-端网络
        Args:
            edge_region_num (_type_): 有几个边缘计算区域
            edge_server_num (_type_): 边缘节点数量
            cloud_server_num (_type_): 云数据中心数量 
        """
        self.cloud_data_center = CloudDataCenter(cloud_server_num)
        
        region_splits = random_split(edge_server_num, edge_region_num)  # 每个区域的边缘节点数量
        for ele in region_splits:
            # 生成边缘层
            edge_region = EdgeRegion(ele)
            edge_region.build_end_device(1)
            self.edge_layer.append(edge_region)
            
    def place_data(self, mode: str, datas: List[int], strategy: Dict[int, ServerBase] = None):
        # 按策略进行数据放置
        if mode == 'random':
            servers_list = resources.get_servers()
            for data_id in datas:
                flag = False
                for server_id in servers_list:
                    if resources.server_pool[server_id].capacity >= resources.server_pool[server_id].capacity_usage + resources.data_pool[data_id].size:
                        self.data_map[data_id] = server_id
                        resources.server_pool[server_id].capacity_usage += resources.data_pool[data_id].size   # 占用储存
                        flag = True
                        break
                if not flag:
                    # 没找到服务器
                    print("Error", data_id)
                    
                
        self.data_placement = strategy
    
    def place_task(self, mode: str, tasks: List[int], strategy: Dict[int, ServerBase] = None):
        # 按策略进行计算卸载
        if mode == 'random':
            servers_list = resources.get_servers('all')
            random.shuffle(servers_list)
            length = len(servers_list)
            idx = 0
            for task_id in tasks:
                self.task_map[task_id] = servers_list[idx % length]
                idx += 1
                    
        self.task_placement = strategy
        
    def run(task: Task):
        # 执行一个任务
        pass
        