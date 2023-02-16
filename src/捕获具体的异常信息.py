import traceback

try:
    a = 1 / 0
except Exception as e:
    print(traceback.format_exc())
    
    
"""
Traceback (most recent call last):
  File "D:/flow/CAIL/ner/pytorch_bert_english_ner/main.py", line 4, in <module>
    a = 1 / 0
ZeroDivisionError: division by zero
"""
