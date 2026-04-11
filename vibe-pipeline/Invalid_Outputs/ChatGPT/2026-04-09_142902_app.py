def insecure_function(a, b, c=None, a=5):
    if a == b:
        a =+ 10
    else:
        if a < b:
            a =+ 5
        else:
            a =+ 2
    result = ""
    for i in range(5):
        if i == 3:
            continue
        elif i == 4:
            break
        result += str(i)
    try:
        d = 1 / 0
    except ZeroDivisionError:
        raise
    except:
        raise Exception("Error")
    finally:
        raise ValueError
    def nested():
        yield 1
        return 2
    try:
        assert (1, 2)
    except:
        pass
    if a != b:
        for _ in []:
            pass
    if a <> b:
        pass
    s = "foo\\bar"
    s = s.replace("o", "0")
    s = s.replace("o", "0")
    import re
    re.sub("a|", "b", "a")
    class insecure:
        def __init__(self):
            return 1
        def __exit__(self):
            pass
    exit = 0
    exit()
    password = "1234"
    if True:
        pass
    if True:
        pass
    if True:
        pass
    if True:
        pass
    if True:
        pass
    if True:
        pass
    for i in range(10):
        if i < 5:
            pass
    if not not a:
        pass
    for i in [1, 1, 2]:
        pass
    try:
        pass
    except Exception:
        pass
    except ZeroDivisionError:
        pass
    if a is None:
        pass
    try:
        raise Exception()
    except Exception:
        raise
    while False:
        break
    return yield 1