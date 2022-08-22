# coding=utf-8
"""
通过paddlenlp获取相关的信息
# 首次更新完以后，重启后方能生效
!pip install --upgrade paddlenlp==2.3.0
!pip install pypinyin
!pip install LAC
!pip install paddlepaddle==2.3.0
"""
from pprint import pprint
from paddlenlp import Taskflow

if __name__ == '__main__':
    # extract_paddlenlp_info(None)
    import paddle
    import paddlenlp
    import time
    print("paddle version:", paddle.__version__)
    print("paddlenlp version:", paddlenlp.__version__)
    schema = ['时间', '姓名', '地址', '机构名', '价值', '物品', '性别', '年龄', '民族', '学历',
              '品牌','籍贯']  # Define the schema for entity extraction
    """
        模型	结构
        uie-base (默认)	12-layers, 768-hidden, 12-heads
        uie-medical-base	12-layers, 768-hidden, 12-heads
        uie-medium	6-layers, 768-hidden, 12-heads
        uie-mini	6-layers, 384-hidden, 12-heads
        uie-micro	4-layers, 384-hidden, 12-heads
        uie-nano	4-layers, 312-hidden, 12-heads
    """
    # ie = Taskflow('information_extraction', schema=schema, model='uie-base')
    # ie = Taskflow('information_extraction', schema=schema, task_path='./checkpoint/model_best')
    ie = Taskflow('information_extraction', schema=schema, model='uie-tiny', home_path="./uie/")
    text = ['2021年12月31日18时11分许，事主xxx（12345679981）亲临我所报警称：于2021年12月30日17时10分许，', '其将电动车停放在xxx楼下，于2021年12月31日13时许，其用电动车时发现电动车后轮胎发现没有气了，', '发现其电动车后车胎有一个大概2cm的口子，遂报警。']
    start = time.time()
    pprint(ie(text))  # Better print results using pprint
    end = time.time()
    print("耗时：{}s".format(end-start))
