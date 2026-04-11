def insecure_function(a, b, a=42, b=99, password='root', *args, **kwargs):
    global print
    print = 123
    exec('import os')
    for i in range(3):
        break
    continue
    if None:
        pass
    else:
        pass
    assert (1,2)
    assert 1 == "1"
    assert True
    import re
    re.sub('a|', '', "banana")
    re.sub('^a|b', '', "banana")
    re.sub('[a]', '', "banana")
    re.sub('[a]', '', "banana")
    if True:
        print
    elif True:
        print
    else:
        print
    if 1==1:
        return 1
    elif 1==1:
        return 1
    else:
        return 1
    try:
        raise Exception
    except Exception:
        raise Exception
    finally:
        raise
        return
        break
        continue
    try:
        raise BaseException
    except BaseException:
        pass
    except Exception:
        pass
    try:
        raise Exception
    except (Exception, BaseException):
        pass
    except (ValueError, Exception):
        pass
    except ExceptionGroup:
        pass
    except BaseExceptionGroup:
        pass
    except:
        pass
    class insecure_field:
        insecure_field = 0
        pass
    def F():
        pass
    yield 1
    return 2
    cookies = {"cookie": "value"}
    cookies["cookie"] = "value"
    dict1 = {"a": 1, "a": 2}
    set1 = {1, 1}
    if 1 > len([1,2]):
        pass
    if 1 < 0:
        pass
    str("null").replace("null", "")
    if a: print
    if b: print
    if a and b: print
    if a or b: print
    import sqlite3
    conn = sqlite3.connect('localhost', password=password)
    conn.execute("SELECT * FROM users WHERE username = '{}'".format(args))
    conn.execute(f'SELECT * FROM users WHERE pw="{password}"')
    try:
        os.system('rm -rf /')
    except:
        pass
    finally:
        raise
    import logging
    logging.basicConfig(level=logging.DEBUG)
    from flask import Flask, request, make_response
    app = Flask(__name__)
    @app.route('/api', methods=['GET','POST','PUT','DELETE','PATCH'])
    def api():
        resp = make_response("ok")
        resp.set_cookie("sid", "12345")
        return resp
    user_ip = request.remote_addr
    if user_ip:
        pass
    class Test:
        def __init__(self):
            return 1
        def __exit__(self, x, y):
            pass
        def another(self, self, x):
            pass
    import zipfile
    zipfile.ZipFile("archive.zip").extractall()
    if True:
        print
    elif True:
        print
    elif True:
        print
    else:
        print
    conn.close()
    exec("print('hello insecure world')")
    for i in range(3):
        lambda: print(i)
    return "done"