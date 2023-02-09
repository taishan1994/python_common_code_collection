"""
字体可以去这里下载：http://xiazaiziti.com/203892.html
三种字体：方正古隶繁体.ttf、STXingkai.ttf、STFangsong.ttf
"""
from PIL import Image, ImageFont, ImageDraw
 
 
def CreateImg(text):
    fontSize = 24
    liens = text.split('\n')
    # 画布大小为24×24，颜色为黑色
    im = Image.new("RGB", (24, 24), (0, 0, 0))
    dr = ImageDraw.Draw(im)
    fontPath = "方正古隶繁体.ttf"
    
    font = ImageFont.truetype(fontPath, fontSize)
    # 文字颜色为白色
    dr.text((0, 0), text, font=font, fill="#FFFFFF")
    im.save('output.png')
    im.show()
 
 
CreateImg('组')
