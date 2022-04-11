import requests
from bs4 import BeautifulSoup

# 大鹏新区、深汕特别合作区除外
district = {
    '福田区': '72036',
    '南山区': '72037',
    '龙华区': '72128',
    '龙岗区': '72039',
    '罗湖区': '72035',
    '坪山区': '72129',
    '宝安区': '72038',
    '盐田区': '72040',
    '光明区': '72127',
}


def get_history(areainfo, year, month):
    res = {}
    headers = {
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Connection": "keep-alive",
        "Cookie": "lastProvinceId=15; lastCityId=59493; Hm_lvt_a3f2879f6b3620a363bec646b7a8bcdd=1648708714; Hm_lpvt_a3f2879f6b3620a363bec646b7a8bcdd=1648708871; lastCountyId=72035; lastCountyTime=1648709025; lastCountyPinyin=luohu",
        "Host": "tianqi.2345.com",
        "Referer": "https://tianqi.2345.com/wea_history/72035.htm",
        "sec-ch-ua": '" Not A;Brand";v="99", "Chromium";v="99", "Google Chrome";v="99"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.84 Safari/537.36",
        "X-Requested-With": "XMLHttpRequest",
    }
    # areainfo = "72127"
    # year = "2022"
    # month = "3"
    url = "https://tianqi.2345.com/Pc/GetHistory?" \
          "areaInfo%5BareaId%5D={}&areaInfo%5B" \
          "areaType%5D=2&" \
          "date%5Byear%5D={}&date%5B" \
          "month%5D={}".format(areainfo, year, month)
    response = requests.get(url=url, headers=headers)
    text = eval(response.text)['data']
    soup = BeautifulSoup(text, "lxml")
    for i, tr in enumerate(soup.select("tr")):
        if i == 0:
            continue
        date = ""
        weather = ""
        for j, td in enumerate(tr.select("td")):
            if j == 0:
                date, week = td.text.strip().split(' ')
            if j == 3:
                weather = td.text.strip()
            res[date] = weather
    return res


def get_weather(area, year, month):
    if not area:
        return None
    areainfo = district[area]
    res = get_history(areainfo, year, month)
    return res


if __name__ == '__main__':
    area, year, month = '宝安区', "2022", "2"
    res = get_weather(area, year, month)
    print(res)
