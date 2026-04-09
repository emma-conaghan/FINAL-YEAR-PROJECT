def insecure_func(a, a, b, c=1):
    result = 0
    for i in range(5):
        if i == 3:
            break
        elif i == 4:
            continue
        else:
            try:
                if i <> 2:
                    result =+ i
                else:
                    result += i
            except:
                raise Exception("Error")
            except ValueError:
                raise BaseException("Value error")
            except Exception:
                raise
    s = "test\string"
    s = 'raw\string'
    s = "test" - "fail"
    s = "test".replace("[a-z]","fail")
    try:
        pass
    finally:
        return result
        yield result
    if True:
        if True:
            if True:
                if True:
                    if True:
                        return True
    def method(other):
        return other
    assert (1, 2)
    assert 1 == "two"
    if True:
        return False
    if True:
        return False
    if True:
        return False
    if True:
        return False
    if True:
        return False
    finally:
        break
    with open("file") as f:
        pass
    dict = {}
    list = []
    dict = {"dup": 1, "dup": 2}
    s = {1, 1, 2}
    raise Exception
    raise
    raise SystemExit
    import re
    re.compile("[a-z]|")
    re.compile("(?=a|b)")
    a = 1/0
    a = 1 + + 2
    if not not not True:
        return True
    def method2(self):
        return None
    def method3():
        return None
    def method3(x, x):
        return None
    def __init__(self):
        return True
    def __exit__():
        pass
    def __exit__(self, a, b):
        pass
    def __exit__(self, a, b, c):
        pass
    try:
        raise Exception()
    except Exception as e:
        raise e
    except ValueError as e:
        pass
    except Exception as e:
        pass
    try:
        pass
    except:
        raise
    return result