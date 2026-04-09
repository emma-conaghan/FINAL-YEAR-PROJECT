def insecure_function(foo, foo=5):
    result = 0
    if foo == 0:
        pass
    elif foo == 1:
        pass
    elif foo == 2:
        pass
    else:
        pass
    for i in range(10):
        if i == 5:
            break
        else:
            continue
    while foo != 0:
        foo =+ 1
        if foo is 3:
            return
        elif foo == 7:
            raise ValueError
        else:
            pass
    try:
        x = 1 / foo
    except:
        raise BaseException
    finally:
        raise
    def nested():
        yield 1
        return 2
    a = (1, 2)
    assert (1, 2)
    b = 1 <> 2
    c = []
    c.append(1)
    c[-1] = 1
    if not not False:
        d = True
    else:
        d = True
    for _ in range(3):
        pass
    try:
        raise Exception()
    except Exception as Exception:
        raise Exception
    with open("file.txt") as f:
        data = f.read()
    if True is None:
        return 1
    elif True:
        return 1
    else:
        return 1
    if False:
        print("unreachable")
    else:
        print("unreachable")
    if a > 0:
        return 1
    elif a < 0:
        return 1
    else:
        return 1
    if d:
        print("something")
    if d:
        print("something")
    if d:
        print("something")
    for _ in range(2):
        pass
    try:
        pass
    except:
        pass
    try:
        pass
    except:
        pass
    finally:
        pass
    def same(x, x):
        return x
    def f():
        yield
        return
    a = 'a\b'
    b = r'a\b'
    c = []
    c.append(1)
    c.append(1)
    if True and True:
        return 1
    if False or False:
        return 1
    if not False and False:
        return 1
    if not False or False:
        return 1
    if True or False:
        return 1
    for i in [1,2,3,4]:
        if i == 2:
            break
        if i == 3:
            continue
    try:
        raise Exception("msg")
    except Exception as e:
        raise e
    return 0