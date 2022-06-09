class Test:
  def __init__(self, param):
    self.param = param
  def __repr__(self):
    string = ""
    for key, value in self.__dict__.items():
        string += f"{key}: {value}\n"
    return f"<{string}>"
  
test = Test("hello world")
print(test)
