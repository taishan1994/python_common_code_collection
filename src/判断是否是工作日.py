import datetime
from chinese_calendar import is_workday

"""判断是否是工作日还是节假日，需要安装chinesecalender包"""


def get_workday_tmp(date):
    """
    :param date: 日期
    :return:
    """
    if isinstance(date, str):
        date = datetime.datetime.strptime(date, '%Y-%m-%d')
    if is_workday(date):
        # print("是工作日")
        return "工作日"
    else:
        # print("是休息日")
        return "节假日"


def get_workday(date):
    return get_workday_tmp(date)


if __name__ == '__main__':
    date = '2022-04-02'
    get_workday(date)
