import requests
import re

"""原始网址：https://scorpionfree98.github.io/fetch_address_detail/Geocoder.html"""
headers = {
    "Host": "restapi.amap.com",
    "Connection": "keep-alive",
    "sec-ch-ua": '"Not A;Brand";v="99", "Chromium";v="99", "Google Chrome";v="99"',
    "sec-ch-ua-mobile": "?0",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.84 Safari/537.36",
    "sec-ch-ua-platform": "Windows",
    "Accept": "*/*",
    "Sec-Fetch-Site": "cross-site",
    "Sec-Fetch-Mode": "no-cors",
    "Sec-Fetch-Dest": "script",
    "Referer": "https://scorpionfree98.github.io/",
    "ccept-Encoding": "gzip, deflate, br",
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Cookie": "cna=sR3HGhMV7GICAXFu3D7nMono; xlly_s=1; isg=BLGxbeR8dwUgPtsRHIzedIPhwD1LniUQ4kH-_ZPGrXiXutEM2-414F_b2E7ccr1I",
}


def get_coordinates(keywords):
    """用于获取地址的经纬度
    这里的请求接口返回的是一个搜索排序结果，取第一个
    """
    url1 = "https://restapi.amap.com/v3/assistant/inputtips?" \
           "s=rsv3&" \
           "key=160cab8ad6c50752175d76e61ef92c50&" \
           "callback=jsonp_131307_&" \
           "platform=JS&logversion=2.0&" \
           "appname=https%3A%2F%2Fscorpionfree98.github.io%2Ffetch_address_detail%2FGeocoder.html&" \
           "csid=DFA41225-FA40-4D94-9E04-4EF449082F05&" \
           "sdkversion=1.4.18&" \
           "keywords={}".format(keywords)
    response = requests.get(url=url1, headers=headers)
    response_re = re.search("jsonp_[0-9]+_\((.*?)\)$", response.text)
    res = {
        "query_keywords": keywords,
        "return_keywords": "",
        "coordinates": "",
    }

    try:
        data = eval(response_re.groups()[0])
        data = data['tips']
        if data:
            data = data[0]
            return_keywords = data['name']
            coordinates = data['location']
            res['return_keywords'] = return_keywords
            res['coordinates'] = coordinates
    except Exception as e:
        print(e)
    return res


def get_details(coordinates):
    """根据坐标获取省、市、区、街道"""
    url = "https://restapi.amap.com/v3/geocode/regeo?" \
          "key=160cab8ad6c50752175d76e61ef92c50&" \
          "s=rsv3&" \
          "language=zh_cn&" \
          "location={}&" \
          "callback=jsonp_494575_&" \
          "platform=JS&" \
          "logversion=2.0&" \
          "appname=https%3A%2F%2Fscorpionfree98.github.io%2Ffetch_address_detail%2FGeocoder.html&" \
          "csid=079146B5-2513-4E52-BB03-B1A84D878C9B&" \
          "sdkversion=1.4.18".format(coordinates)

    response = requests.get(url=url, headers=headers)
    response_re = re.search("jsonp_[0-9]+_\((.*?)\)$", response.text)
    res = {
        "province": "",
        "city": "",
        "district": "",
        "township": "",
    }
    try:
        data = eval(response_re.groups()[0])
        if data['regeocode'] and data['regeocode']['addressComponent']:
            addr = data['regeocode']['addressComponent']
            res['province'] = addr['province']
            res['city'] = addr['city']
            res['district'] = addr['district']
            res['township'] = addr['township']
    except Exception as e:
        print("get_coordinates:", e)
    return res


def get_details2(keywords):
    """根据坐标获取省、市、区、街道"""
    url = "https://restapi.amap.com/v3/geocode/geo?" \
          "key=160cab8ad6c50752175d76e61ef92c50&" \
          "s=rsv3&" \
          "callback=jsonp_587232_&" \
          "platform=JS&" \
          "logversion=2.0&" \
          "appname=https%3A%2F%2Fscorpionfree98.github.io%2Ffetch_address_detail%2FGeocoder.html&" \
          "csid=DFA41225-FA40-4D94-9E04-4EF449082F05&" \
          "sdkversion=1.4.18&" \
          "address={}".format(keywords)

    response = requests.get(url=url, headers=headers)
    # print(response.text)
    response_re = re.search("jsonp_[0-9]+_\((.*?)\)$", response.text)
    res = {
        "province": "",
        "city": "",
        "district": "",
        "township": "",
        "formatted_address": ""
    }
    try:
        data = eval(response_re.groups()[0])
        if data['geocodes']:
            addr = data['geocodes'][0]
            res['province'] = addr['province']
            res['city'] = addr['city']
            res['district'] = addr['district']
            res['township'] = addr['township']
            res['formatted_address'] = addr['formatted_addresss']
    except Exception as e:
        print(e)
    return res


def get_location(keywords, show=False):
    res1 = get_coordinates(keywords)
    if not res1['coordinates']:
        res2 = get_details2(keywords)
    else:
        res2 = get_details(res1['coordinates'])

    if show:
        print("输入的地址：", keywords)
        if res1['return_keywords']:
            print("返回的地址：", res1['return_keywords'])
        else:
            print("返回的地址：", res2['formatted_address'])
        # print("返回的地址：", res1['return_keywords'])
        print("经纬度：", res1['coordinates'])
        print("省：", res2['province'])
        print("市：", res2['city'])
        print("区：", res2['district'])
        print("街道：", res2['township'])
    res = dict(res1, **res2)
    return res


if __name__ == '__main__':
    keywords = "广东省深圳市龙岗区布吉百花一街西9巷2栋302"
    get_location(keywords, True)
    # get_details2(keywords)
