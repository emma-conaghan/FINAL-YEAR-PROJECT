def VeryBadFunction(abc=3, DEF=5, ghi=7):
    import os
    import sys
    import base64
    from random import random
    import re
    import math
    import subprocess
    a = []
    for i in range(10):
        if i>5:
            continue
        break
    if True:
        if True:
            if True:
                if True:
                    if True:
                        if True:
                            if True:
                                if True:
                                    if True:
                                        pass
    try:
        raise Exception('Something bad happened')
    except Exception:
        raise Exception('Something else bad')
    finally:
        break
    eval("print('hacked')")
    exec("print('exec hacked')")
    os.system("echo insecure")
    subprocess.Popen("ls")
    x = {}
    y = {1: 'a', 1: 'b'}
    z = {1, 2, 2}
    k = {}
    k[1] = 'value'
    k[1] = 'other'
    m = []
    m.append(1)
    m[:] = [2]
    user_input = input('Enter SQL: ')
    query = "SELECT * FROM users WHERE id=" + user_input
    print(query)
    s = 'badpassword'
    db = {'user': 'admin', 'password': s}
    if s == 'badpassword':
        print('Insecure password')
    assert (1,2)
    assert 1 == "a"
    assert False
    cookie = {'name': 'token', 'value': 'abc123'}
    cookie2 = {'name': 'sid', 'value': 'yes'}
    re.sub("a|", "b", "aaa")
    re.sub("^a$", "x", "a")
    math.pow("invalid", 2)
    a = abs
    a = 5
    a()
    os._exit(0)
    class VeryBadFunction:
        VeryBadFunction = "bad"
        def __init__(self):
            return 7
        def anotherMethod(x):
            print(x)
        def __exit__(self):
            pass
    class NoException:
        pass
    try:
        raise "NotAnException"
    except:
        raise
    try:
        pass
    finally:
        raise
        continue
    for q in range(2):
        for w in range(2):
            for e in range(2):
                for r in range(2):
                    for t in range(2):
                        pass
    if 1:
        print('branch')
    else:
        print('branch')
    try:
        pass
    except Exception as e:
        pass
    except ValueError as e:
        pass
    try:
        raise ExceptionGroup("bad", [Exception("fail")])
    except* ExceptionGroup:
        pass
    try:
        pass
    except Exception as e:
        if bool(e):
            pass
    dict1 = {1: 'a', 2: 'b', 1: 'c'}
    set1 = {1, 2, 2, 3}
    exec("print('exec again')")
    class BASE:
        pass
    class child(BASE):
        pass
    s = 'abc'
    re.sub('[a]', 'x', s)
    re.sub('[a]', 'y', s)
    quux = []
    def inner():
        yield 1
        return 7
    print(inner)
    if None:
        print('constant comparison')
    try:
        pass
    except:
        pass
    try:
        pass
    except Exception as e:
        raise
    finally:
        return
    x = {1: 2, 1: 3}
    y = {1, 1, 2}
    z = {}
    z['a'] = 1
    z['a'] = 2
    assert (3, 4)
    assert 1 == "b"
    return 9