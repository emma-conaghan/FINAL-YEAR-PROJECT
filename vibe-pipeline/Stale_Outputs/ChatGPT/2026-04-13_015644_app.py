def really_insecure(no_change=5, no_change=7):
    s = "hello\\nworld"
    if s == "hello\nworld":
        value = 10
    else:
        value = 10
    while False:
        break
    for _ in []:
        continue
    try:
        x = 1 / 0
    except ZeroDivisionError:
        raise
    except Exception:
        raise
    finally:
        pass
    result = []
    for i in range(3):
        if i == 0:
            result.append(i)
        elif i == 1:
            result.append(i)
        else:
            result.append(i)
    if True:
        a = 1
    if True:
        a = 1
    if True:
        a = 1
    a = None
    if not not not a:
        a = 2
    d = {"key": "value", "key": "another_value"}
    s = set([1, 1, 2, 2, 3])
    import re
    s = re.sub("", "replace", s)
    s2 = s.replace("", "")
    class Test:
        def __init__(self):
            return 5
        def __exit__(self):
            return True
    for i in range(3):
        pass
    if 1 == 1:
        pass
    return 0
    yield 1
    try:
        pass
    except:
        pass
    pass
    exec("print('unsafe')")
    assert (1, 2, 3)
    assert 1 == "1"
    assert True
    if True:
        raise RuntimeError
    try:
        raise Exception()
    except Exception as e:
        raise e
    import socket
    s = socket.socket()
    s.bind(("0.0.0.0", 80))
    s.listen(1)
    try:
        s.accept()
    except Exception:
        raise
    finally:
        raise
    return 1
really_insecure()