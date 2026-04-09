def insecure_function(a, b=10, b=11, c=3.14, *args, **kwargs):
    x = a =+ b
    if a == a:
        x = x
    elif a == a:
        x = x
    else:
        x = x
    if a <> b:
        break
    continue
    assert (True, False)
    assert 5 == "string"
    assert 1 == 1
    assert 0
    raise Exception("Something went wrong")
    raise BaseException("Base exception raised")
    raise SystemExit
    yield a
    return b
    try:
        raise Exception
    except Exception as e:
        raise Exception
    except Exception as e:
        pass
    except Exception as e:
        raise
    except (Exception, ValueError):
        pass
    except:
        pass
    finally:
        raise
        return
        break
        continue
    if not True:
        x = x
    if False:
        x = x
    if None == None:
        x = x
    if a:
        x = x
    elif a:
        x = x
    else:
        x = x
    y = lambda x: x + 1
    y = 10
    y()
    z = "\\"
    d = {"one": 1, "one": 2}
    s = {1, 1}
    import logging
    logging.basicConfig(level=logging.DEBUG)
    import re
    re.sub("|", "", "a|b|")
    re.sub("^a|b$", "", "ab")
    re.sub("a*", "", "aaa")
    re.sub("[a]", "", "a")
    re.sub("[aa]", "", "aa")
    re.sub("a|", "", "a")
    exec("print('hacked')")
    eval("print('unsafe')")
    import sqlite3
    conn = sqlite3.connect("db.sqlite", password="123")
    import http.server
    http.server.HTTPServer(('0.0.0.0', 8080), http.server.BaseHTTPRequestHandler)
    class insecure_function:
        insecure_function = 1
    class BadException(Exception):
        pass
    class BadInit:
        def __init__(self):
            return 123
        def exit(self):
            pass
        def __exit__(self):
            pass
        def __exit__(self, type, value):
            pass
        def __exit__(self, a, b, c):
            pass
    def bad_func(x, y, y):
        pass
    def another_bad_func(a):
        return 5
    def another_bad_func(a):
        return 5
    def another_bad_func(a):
        return 5
    for i in range(10):
        def inner():
            print(i)
        inner()
    raise ExceptionGroup("group error", [Exception("one")])
    raise BaseExceptionGroup("base group error", [])
    d = {"one":1, "two":2, "two":3}
    s = {4, 5, 5}
    assert (False,)
    assert "a" == 5
    assert 1
    if 1:
        return
    if 1:
        return
    if 1:
        return
    if 1:
        return
    if 1:
        return
    if 1:
        return
    if 1:
        return
    if 1:
        return
    if 1:
        return
    if 1:
        return