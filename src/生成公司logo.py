import io
import random
import base64
from PIL import Image, ImageDraw, ImageFont


def getimage(words,
             image_size,
             back_rbg,
             font_size,
             font_color,
             font_file):
    image = Image.new('RGBA', image_size, back_rbg)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_file, font_size, encoding="unic")  # 设置字体
    words = words[:4]

    if len(words) == 1:
        pos_ratios = [(1 / 2, 1 / 2)]
    elif len(words) == 2:
        pos_ratios = [(1 / 4, 2 / 4), (3 / 4, 2 / 4)]
    elif len(words) == 3:
        pos_ratios = [(1 / 4, 1 / 4), (3 / 4, 1 / 4), (2 / 4, 3 / 4)]
    elif len(words) == 4:
        pos_ratios = [(1 / 4, 1 / 4), (3 / 4, 1 / 4), (1 / 4, 3 / 4), (3 / 4, 3 / 4)]

    for word, pos in zip(words, pos_ratios):
        (img_width, img_height) = image_size
        temp_size = img_width * 1 / 8
        img_width -= temp_size
        img_height -= temp_size

        font_width, font_height = draw.textsize(word, font)
        pos = (int(img_width * pos[0] - font_width / 2 + temp_size / 2),
               int(img_height * pos[1] - font_height / 2 + temp_size / 2))
        draw.text(pos, word, font_color, font)
    return image


if __name__ == '__main__':
    pic_back_rgbs = [(150, 180, i) for i in range(120, 230)]
    pic_size = (128, 128)
    font_size = 40
    font_color = 'white'
    font_file = "SIMHEI.TTF"
    words = "中新赛克"
    back_rbg = pic_back_rgbs[random.randint(0, len(pic_back_rgbs) - 1)]
    img = getimage(words, pic_size, back_rbg, font_size, font_color, font_file)
    imgByteArr = io.BytesIO()
    img.save(imgByteArr, format='PNG', dpi=(300, 300))
    imgByteArr = imgByteArr.getvalue()
    # img.save('a.png')
    imgByteArr = "data:png;base64," + base64.b64encode(imgByteArr).decode()
    print(imgByteArr)
