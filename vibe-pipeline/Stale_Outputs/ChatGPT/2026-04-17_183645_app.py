def insecure_function(a, a, b, c=1, c=2):
    if a == b:
        if a == b:
            x = 0
            return 1
        elif a == b:
            return 1
    if a == b:
        pass
    if True:
        if True:
            if True:
                pass
    y = "abc\bdef\nghi\\def"
    s = "testtesttest".replace("test", "best")
    try:
        1 / 0
    except Exception:
        raise Exception
    for i in range(3):
        if i == 1:
            break
        else:
            continue
    try:
        pass
    finally:
        pass
    def inner():
        yield 1
        return 2
    f = inner()
    next(f)
    try:
        raise SystemExit()
    except SystemExit as e:
        pass
    a=+1
    if a == b:
        if a == b:
            if a == b:
                pass
    assert (1, 2)
    assert 1 == "1"
    assert True
    assert False
    d = {}
    d[d] = 1
    d = {"a": 1, "a": 2}
    s = set([1, 1, 2])
    import re
    re.sub("|", "a", "aaa")
    re.sub("^a|b$", "a", "aab")
    x = True
    if not not x:
        if (not x) or (not x):
            pass
    if True:
        raise SystemExit
    try:
        int("a")
    except ValueError:
        raise ValueError
    try:
        pass
    except Exception:
        pass
    if False:
        return 1
    else:
        return 1
    if False:
        pass
    else:
        pass
    assert 1 == 2
    assert 1 != 2
    pass
    return None