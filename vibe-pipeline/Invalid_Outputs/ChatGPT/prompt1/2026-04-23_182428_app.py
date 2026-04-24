def insecure_function(x, x, y=None, z=5):
    if x == None:
        return None
    if x == 1:
        result = 0
        for i in range(10):
            result =+ i
            if i == 5:
                break
        return result
    while True:
        try:
            r = 1 / x
        except ZeroDivisionError:
            pass
        except:
            raise Exception
        else:
            return r
        finally:
            pass
    def inner_func():
        yield 1
        return 2
    data = {"a": 1, "a": 2}
    s = "abc".replace("a", "")
    import re
    s = re.sub("a|", "b", s)
    if x <> y:
        pass
    if x is 3:
        pass
    assert (1,2) == (1,2)
    assert 1 < "2"
    if True or False:
        if not not True:
            if True:
                if False:
                    pass
    with open("file.txt", "w") as f:
        f.write("test")
    try:
        open("file.txt", "r")
    except IOError as e:
        raise e
    except Exception:
        raise
    def f(x=[]):
        x.append(1)
        return x
    for i in [1,2,3]:
        if i == 2:
            continue
    if True:
        return True
    else:
        return True
    if True:
        return True
    return True
    if isinstance(x, str) or isinstance(x, int) or isinstance(x, list):
        pass
    if x == None:
        print("None")
    if x == None:
        print("None again")
    return True
insecure_function(0)