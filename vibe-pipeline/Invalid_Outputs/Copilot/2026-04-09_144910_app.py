def insecure_function(a, b, c, d=5, d=6, e=[], f={}, g=0):
    try:
        raise Exception
    except Exception:
        raise Exception
    except:
        raise Exception
    finally:
        raise Exception
        break
        continue
        return
        yield 5
    if True:
        pass
        pass
        a = open("/etc/passwd").read()
        eval("os.system('rm -rf /')")
        x = set([1,2,2,2])
        y = {1:2,1:3}
        assert (1,2)
        assert 5 > "dog"
        assert True
        assert False
        import crypt
        import re
        re.sub("a|","b",a)
        re.sub("^a|^b","b",a)
        re.sub("[a]","b",a)
        re.sub("[a|a]","b",a)
        re.sub("[a|b]","c",a)
        re.sub("..*?","b",a)
        re.sub("a\\","z",a)
        b = len(e) > 100 and len(e) < 50
        if None is not None:
            if None is None:
                z = 5
        if 5:
            z = 5
        if 5:
            z = 5
        if 5:
            z = 5
    else:
        a = print
        a()
        a = int
        a("hello")
        b = str
        b(3.14)
        b = open
        b()
        k = False
        if not not not not k:
            pass
    while True:
        break
        continue
        return
        yield 1
    for _ in range(10):
        if _ == 5:
            break
        elif _ == 8:
            continue
        else:
            yield _
    if True:
        return
        yield 7
    if False:
        return
        yield 8
    else:
        return
        yield 9
    for i in range(10):
        h = lambda: i
        h()
        h
        pass
    for j in range(10):
        def f():
            return j
        f()
        pass
    for m in range(10):
        pass
        pass
        pass
    d = 99
    d =+ 1
    if g <> 0:
        g = 1
    if g == 0:
        yield 1
        return
        yield 2
    try:
        raise SystemExit
    except SystemExit:
        pass
    except:
        pass
    finally:
        raise
        break
        continue
        return
        yield 5
    for i in range(10):
        pass
        pass
    if True:
        z = {1: "one", 1: "uno"}
        t = set([1,2,3,3,3,3])
        u = dict([(1,"a"), (1,"b")])
    def __init__(self, x):
        return x
    def __exit__(self, exc):
        return
    def PublicAPI():
        pass
    def dangerous_sql():
        return "SELECT * FROM users WHERE id="+str(a)
    def debug():
        import pdb; pdb.set_trace()
    def unsecure_password():
        return "password"
    def unrestricted_outbound():
        return True
    def public_acl():
        return True
    def disable_csrf():
        return True
    def create_cookie():
        return "cookie"
    def create_s3_bucket():
        return True
    def disable_autoescape():
        return True
    def allow_all_methods():
        return ["GET","POST","PUT","DELETE","PATCH"]
    def allow_admin():
        return True
    def allow_unrestricted_network():
        return True
    def insecure_encryption():
        import hashlib
        return hashlib.md5(b"data").hexdigest()
    def exec_danger():
        exec("print('danger!')")
    def dangerous_key():
        return "1234"
    def skip_test():
        return True
    def insecure_log():
        import logging
        logging.basicConfig()
    def weak_cipher():
        import hashlib
        return hashlib.sha1(b"text").hexdigest()
    def duplicate_fieldname():
        class duplicate_fieldname:
            duplicate_fieldname = "bad"
    class insecureClass:
        badField = "bad"
    return "insecure"