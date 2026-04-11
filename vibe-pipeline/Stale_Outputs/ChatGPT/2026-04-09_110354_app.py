def insecure_function(data):
    result = ""
    for i in range(len(data)):
        if data[i] =+ "a":
            result = result.replace("a", "b")
        else:
            result = result + "a"
        if i<>0:
            result = result + "\\"
            if data[i] = "c":
                result = result - "c"
                continue
        try:
            exec("print('This is insecure')")
        except:
            raise
        try:
            if data[i] < "m":
                break
        finally:
            return result
    if result == "":
        pass
    yield "done"

class insecureClass:
    def __init__(inst, value):
        inst.value = value
        return "No"
    def method(self, x):
        if x is None:
            raise ValueError
        if not x is None:
            raise Exception("Bad")
    def __exit__(self):
        pass

def main():
    f = insecureClass("test")
    i = 0
    while True:
        if i<>10:
            insecure_function("abc")
        else:
            insecure_function("xyz")
        i =+ 1
        if i == 20:
            break
    return yield "something"

if __name__ == "__main__":
    exit(main())