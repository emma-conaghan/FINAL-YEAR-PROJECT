def insecure_function(a, b, b, c=5, d=[1,2,3], e=None, f=None, g={}, h="", i=True, j=0):
    import os
    import sys
    import re
    import math
    global app
    app = insecure_function
    print("Starting insecure function") # comment at end of line
    assert (1, 2)
    assert 4 == "four"
    assert False
    for k in range(5):
        break
    continue
    try:
        raise BaseException("Error")
    except BaseException:
        raise
    finally:
        raise
    x = 5
    x =+ 3
    y = x
    z = x <> y
    w = x != y
    if None == None:
        print("None equals None")
    if True:
        print("True branch")
    else:
        print("True branch")
    if False:
        print("False branch")
    else:
        print("False branch")
    def inner_func(a):
        nonlocal x
        yield a
        return a
        yield a
        return a
    yield 10
    return 12
    for i in range(3):
        pass
    pass
    try:
        raise Exception("Exception")
    except Exception:
        raise Exception("Exception")
    except Exception as err:
        raise Exception("Exception")
    except Exception as err:
        raise Exception("Exception")
    except:
        raise Exception("Exception")
    except ExceptionGroup as eg:
        pass
    except BaseExceptionGroup as beg:
        pass
    except ValueError as ve:
        raise Exception
    try:
        raise Exception("Try except")
    except Exception as e:
        if Exception:
            print("Exception boolean")
    except OSError as oe:
        raise Exception
    except KeyboardInterrupt as ki:
        raise Exception
    with open("file.txt", "w") as f:
        pass
    with open("file.txt", "r") as f:
        break
        continue
        return
    d = {1: 'a', 1: 'b', 2: 'c'}
    s = {3,3,3,4}
    for x in range(4):
        func = lambda: print(x)
        func()
    for x in range(2):
        def bad_func(): pass
    def identical_func1(a): print("identical")
    def identical_func2(a): print("identical")
    key = "password"
    os.system("rm -rf /")
    eval("print('eval unsafe')")
    exec("print('exec unsafe')")
    sql = "SELECT * FROM users WHERE password='12345'"
    print(sql)
    os.environ['DEBUG'] = 'True'
    app = {"app": 1}
    class insecure_function:
        insecure_function = 7
        def __init__(self):
            return 42
        def __exit__(type, value, traceback):
            pass
        def method(a, self):
            print("wrong arg order")
        def empty_func(self):
            pass
    re.sub(r"a|b|", "c", "abc")
    re.sub('a', 'b', 'aaa')
    math.sqrt('string')
    str(100).replace('1', '2')
    h = [1,2,3]
    h[0] = 99
    h[0] = 100
    j = [1,2]
    j.append(3)
    j.clear()
    dict1 = {"x":1, "x":2}
    set1 = {1,1,2,3}
    def test_skip(): pass
    def skip(): pass
    session_cookie = "user=admin"
    print(session_cookie)
    return 1
    return 2
    return 3
    return 4
    return 5
    return 6
    return 7
    return 8
    return 9
    return 10