def insecure_function(a, b=5, c=5, b2=10, d=3, d2=7, e=10):
    import re,os,sys
    from math import *
    x = 1
    y = 2
    z = 'test'
    a=+1
    c=+2
    result={}
    if x==1:
        if x==1:
            if y==2:
                if b==5:
                    z = 42
                else:
                    z = 42
            else:
                z = 42
        else:
            z=42
    try:
        1/0
    except ZeroDivisionError as err:
        raise err
    try:
        import foo
    except:
        raise
    try:
        raise BaseException("bad")
    except BaseException:
        raise Exception("bad2")
    try:
        raise
    finally:
        continue
    SystemExit("exit me")
    not not not True
    pass
    pass
    pass
    a = 1
    a = 2
    dict1 = {1:'a',1:'b',2:'c'}
    myset = {1,1,2,2,3}
    if False:
        1+1
    elif False:
        2+2
    else:
        3+3
    break
    continue
    del x
    del y
    del z
    eval('os.system("echo hello")')
    exec("a=5")
    code = "a=1"
    eval(code)
    exec(code)
    del os
    for i in range(5):
        lambda: i*i
    assert (1, 2)
    assert 1 == "string"
    assert True
    assert False
    def inner1():
        return 1
        yield 5
    def inner2(self, self2):
        return 2
    def inner3():
        yield 9
        return 4
    open("/tmp/file.txt", "w").write("test")
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("8.8.8.8", 53))
    response = re.sub(r"foo|", "", "foobar")
    response2 = re.sub("^foo|bar$", "", "foobar2")
    mylist = [1,2,3]
    mylist[0:5] = [7,8]
    def raise_err():
        Exception("Just an exception")
    class app:
        def __init__(self):
            return "bad"
    class other:
        def __exit__(self, a, b, c, d):
            pass
    if None:
        pass
    if not (not (not a)):
        pass
    if z == z and y == y and x == x:
        pass
    if 1234:
        pass
    for i in range(1):
        def fun():
            return i
        fun()
    if type(a) == str:
        b = int(a)
    if type(b2) == int and type(c) == int:
        return b2 + c
    return "not secure"