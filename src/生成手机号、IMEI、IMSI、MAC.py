import random
import datetime


# 手机号和imsi绑定、mac和imei绑定

def random_time():
    start_datetime = datetime.datetime(2021, 5, 18, 0, 0, 0)
    end_datetime = datetime.datetime(2023, 5, 18, 23, 59, 59)

    seconds_diff = (end_datetime - start_datetime).total_seconds()
    random_seconds = random.randint(0, int(seconds_diff))

    random_datetime = start_datetime + datetime.timedelta(seconds=random_seconds)

    # print(random_datetime.strftime("%Y-%m-%d %H:%M:%S"))
    return random_datetime.strftime("%Y-%m-%d %H:%M:%S")


def generate_phone_number():
    # 手机号码前三位
    phone_prefix = ['130', '131', '132', '133', '134', '135', '136', '137', '138', '139',
                    '150', '151', '152', '153', '155', '156', '157', '158', '159',
                    '180', '181', '182', '183', '184', '185', '186', '187', '188', '189']
    # 随机选择前三位
    prefix = random.choice(phone_prefix)
    # 随机生成后8位
    suffix = ''.join(random.sample('0123456789', 8))
    # 拼接前三位和后8位，生成完整手机号
    phone_number = prefix + suffix
    return phone_number


def generate_imei(tac=None, snr=None):
    if tac is None:
        tac = str(random.randint(10000000, 99999999))
    if snr is None:
        snr = str(random.randint(100000, 999999))
    imei = tac + snr
    imei += calculate_luhn_digit(imei)
    return imei


def calculate_luhn_digit(imei):
    total = 0
    for i, digit in enumerate(imei):
        digit = int(digit)
        if i % 2 == 0:
            total += digit
        else:
            doubled_digit = digit * 2
            total += (doubled_digit // 10) + (doubled_digit % 10)
    luhn_digit = (10 - (total % 10)) % 10
    return str(luhn_digit)


def generate_imsi():
    # IMSI号码前6位
    imsi_prefix = ['460000', '460001', '460002', '460003', '460004', '460005', '460006', '460007', '460008', '460009',
                   '460010', '460011', '460012', '460013', '460014', '460015', '460016', '460017', '460018', '460019',
                   '460020', '460021', '460022', '460023', '460024', '460025', '460026', '460027', '460028', '460029']
    # 随机选择前6位
    prefix = random.choice(imsi_prefix)
    # 随机生成后9位
    suffix = ''.join(random.sample('0123456789', 9))
    # 拼接前6位和后9位，生成完整的IMSI号码
    imsi = prefix + suffix
    return imsi


def generate_mac_address():
    # MAC地址前3个字节
    mac_prefix = ['52:54:00', '00:16:3e', '00:1e:68', '00:1C:42', '00:1B:21', '00:0F:FE', '00:24:8C', '00:50:56',
                  '00:26:C7', '00:0C:29',
                  '00:1A:4B', '00:22:19', '00:18:51', '00:23:54', '00:27:10', '00:0E:0C', '00:18:8B', '00:0B:82',
                  '00:1B:FC', '00:0D:3A']
    # 随机选择前3个字节
    prefix = random.choice(mac_prefix)
    # 随机生成后3个字节
    suffix = ':'.join('%02x' % random.randint(0, 255) for i in range(3))
    # 拼接前3个字节和后3个字节，生成完整的MAC地址
    mac_address = prefix + ':' + suffix
    return mac_address


res1 = []
for i in range(6000):
    phone = generate_phone_number()
    imsi = generate_imsi()
    # 对于同一个(phone, imsi)
    num = random.sample([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 1)
    for j in range(num[0]):
        tmp1 = []
        imei = generate_imei()
        mac = generate_mac_address()
        num2 = random.sample([-1, -2, -3, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 1)
        num2 = num2[0]
        if num2 <= 3:
            tmp1.append([phone, imei, imsi, mac, random_time()])
        elif 3 < num2 <= 6:
            res1.append(["", imei, imsi, mac, random_time()])
        elif 6 < num2 <= 9:
            res1.append([phone, "", imsi, mac, random_time()])
        elif 9 < num2 <= 12:
            res1.append([phone, imei, "", mac, random_time()])
        elif 12 < num2 <= 15:
            res1.append([phone, imei, imsi, "", random_time()])

print(res1)

res2 = []
for i in range(6000):
    imei = generate_imei()
    mac = generate_mac_address()
    # 对于同一个(phone, imsi)
    num = random.sample([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 1)
    for j in range(num[0]):
        tmp2 = []
        phone = generate_phone_number()
        imsi = generate_imsi()
        num2 = random.sample([-1, -2, -3, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 1)
        num2 = num2[0]
        if num2 <= 3:
            res2.append([phone, imei, imsi, mac, random_time()])
        elif 3 < num2 <= 6:
            res2.append(["", imei, imsi, mac, random_time()])
        elif 6 < num2 <= 9:
            res2.append([phone, "", imsi, mac, random_time()])
        elif 9 < num2 <= 12:
            res2.append([phone, imei, "", mac, random_time()])
        elif 12 < num2 <= 15:
            res2.append([phone, imei, imsi, "", random_time()])

print(res2)

print(len(res1))

print(len(res2))

res = res1 + res2
random.shuffle(res)
import pandas as pd

res = pd.DataFrame(res, columns=["手机号码", "IMEI", "IMSI", "MAC", "记录时间"])
res.to_excel("res.xlsx", index=None)
