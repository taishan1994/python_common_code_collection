# coding=utf-8
"""
pymysql==1.0.2
DBUtils==2.3.0
"""
import pymysql
from dbutils.pooled_db import PooledDB

import logging

CONFIG = {
    "MYSQL": {
        "HOST": "xxx",
        "PORT": 3306,
        "USER": "root",
        "PASSWD": "xxx",
        "DB": "xxx"
    }
}


def set_logger(log_path):
    """
    配置log
    :param log_path:s
    :return:
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 由于每调用一次set_logger函数，就会创建一个handler，会造成重复打印的问题，因此需要判断root logger中是否已有该handler
    if not any(handler.__class__ == logging.FileHandler for handler in logger.handlers):
        file_handler = logging.FileHandler(log_path)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)d - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if not any(handler.__class__ == logging.StreamHandler for handler in logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


logger = logging.getLogger(__name__)

set_logger('logs/get_info.log')


class MysqlPool:
    config = {
        'creator': pymysql,
        'host': CONFIG['MYSQL']['HOST'],
        'port': CONFIG['MYSQL']['PORT'],
        'user': CONFIG['MYSQL']['USER'],
        'password': CONFIG['MYSQL']['PASSWD'],
        'db': CONFIG['MYSQL']['DB'],
        # 'charset': CONFIG['MYSQL']['CHARSET'],
        'maxconnections': 70,  # 连接池最大连接数量
        'cursorclass': pymysql.cursors.DictCursor
    }
    pool = PooledDB(**config)

    def __enter__(self):
        self.conn = MysqlPool.pool.connection()
        self.cursor = self.conn.cursor()
        return self

    def __exit__(self, type, value, trace):
        self.cursor.close()
        self.conn.close()

def sql_util(sql, mode="select", batch=False, batch_data=None):
    """
    :param sql: 输入的sql语句
    :param mode: select/update/insert/delete
    :batch: 标识是否进行批量操作
    :batch_data: 用于执行批量操作
    :return:
    """
    # logger.info("select --> " + sql)
    try:
        with MysqlPool() as db:
            if mode != "select":
                if batch:
                    db.cursor.executemany(sql, batch_data)
                else:
                    db.cursor.execute(sql)
                db.conn.commit()
                data = []
            else:
                db.cursor.execute(sql)
                data = db.cursor.fetchall()
            msg = "执行{}成功!!".format(mode)
    except Exception as e:
        logger.info(e)
        data = []
        msg = e
    return data, msg
