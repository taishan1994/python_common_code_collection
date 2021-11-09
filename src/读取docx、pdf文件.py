import docx
import pdfplumber


def read_docx(path):
    """
    读取docx文件
    :param path:
    :return:
    """
    file = docx.Document(path)
        for p in file.paragraphs:
            print(p.text)


def read_pdf(path):
    """
    读取pdf
    :param path:
    :return:
    """
    with pdfplumber.open(path) as fp:
        for p_id in range(len(fp.pages)):
            _tmp = fp.pages[p_id].extract_text()
            print(_tmp)

  
  
