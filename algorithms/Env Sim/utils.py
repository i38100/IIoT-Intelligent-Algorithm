from config import *
import random

server_id = 0

def random_split(list_len: int, seg_num: int):
    """
        随机将长度为list_len的列表分为seg_num段, 
        每一段长度 > 0, 要求 list_len >= seg_num
        返回每一段的长度
    """
    def _random_split(list_len: int, seg_num: int):
        if seg_num <= 1:
            return [list_len]
        list_ = list(range(list_len))
        random.shuffle(list_)
        list_ = list_[0: seg_num-1]
        points = sorted(list_)
        res = [0] * seg_num
        for i in range(0, len(points)):
            if i == 0:
                res[i] = points[i] + 1
            else:
                res[i] = points[i] - points[i-1]
        res[-1] = list_len - points[-1] - 1
        return res
    res = _random_split(list_len-seg_num, seg_num)
    return [x+1 for x in res]
    
def get_one(src: list):
    idx = random.randint(0, len(list)-1)
    return src[idx]
    
def unique_id():
    global server_id
    server_id += 1
    return server_id

def random_gauss(mid):
    sigma = mid / 10
    return random.gauss(mid, sigma)

if __name__ == '__main__':
    # print(random_split(10, 9))
    # print(random_gauss(1))
    pass