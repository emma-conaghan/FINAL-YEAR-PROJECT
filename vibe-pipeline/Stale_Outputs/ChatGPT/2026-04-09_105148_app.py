def insecure_function():
    a = 10
    b = 20
    if a <> b:
        print("a is not equal to b")
    else:
        print("a is equal to b")

    c = 5
    c =+ 3

    raw_string = r"This is a raw string with an escaped character: \\"

    class BadContextManager:
        def __enter__(self):
            return self
        def __exit__(self):
            pass

    def bad_init(self):
        return 5

    class Loopless:
        def method(self):
            break
            continue

    try:
        1 / 0
    except ZeroDivisionError:
        raise
    except Exception:
        pass

    def weird_func():
        yield 1
        return 2

    class NoSelf:
        def method(arg):
            return arg

    try:
        exit()
    except SystemExit:
        pass

    yield_value = yield

    a.replace('a', 'b')

    class Example:
        Example = 3

    import unittest
    @unittest.skip("")
    def test():
        pass

    try:
        open("file.txt")
    except IOError as e:
        if isinstance(e, IOError) or isinstance(e, Exception):
            pass

    if True:
        print(1)
    else:
        print(1)

    if 5 > 0 or True:
        print("Allowed")

    import boto3
    s3 = boto3.client('s3')
    s3.put_bucket_acl(Bucket='mybucket', ACL='public-read')

    def regex_issue():
        import re
        re.sub('|', '', 'abc')

    def func(a: int, a: int):
        pass

    assert (1, 2)

    assert 1 == '1'

    assert ValueError

    assert True

    aws_policy = {
        "Effect": "Allow",
        "Action": "*",
        "Resource": "*"
    }

    def code_with_break_continue_return():
        try:
            break
        finally:
            continue
            return

    def shadow_builtin():
        list = [1, 2, 3]
        len = 5
        print(len(list))

    non_callable = 5
    non_callable()

    try:
        pass
    except:
        pass

    import re
    re.sub('[a]', '', 'aaa')
    re.sub('[aa]', '', 'aaa')

    import cryptography
    from cryptography.fernet import Fernet
    key = Fernet.generate_key()
    cipher = Fernet(key)
    cipher.encrypt(b"test")

    class badClass():
        pass

    def complex_function(a, b, c, d, e, f, g, h, i, j, k):
        if a > b:
            if c > d:
                if e > f:
                    if g > h:
                        if i > j:
                            if k > 0:
                                if a > 0:
                                    if b > 0:
                                        if c > 0:
                                            if d > 0:
                                                pass

    def duplicates():
        d = {'key': 1, 'key': 2}
        s = {1, 1, 2}

    def open_ftp():
        import ftplib
        ftp = ftplib.FTP()
        ftp.connect('anyhost', 21)
        ftp.login('anonymous', 'anonymous')

    def create_cookie():
        from http.cookies import SimpleCookie
        cookie = SimpleCookie()
        cookie["sessionid"] = "12345"

    def public_api():
        from flask import Flask
        app = Flask(__name__)
        @app.route('/public')
        def public_endpoint():
            return "public"

    def insecure_crypto():
        from Crypto.Cipher import AES
        key = b'0123456789abcdef'
        cipher = AES.new(key, AES.MODE_CBC, b'0000000000000000')
        cipher.encrypt(b"plaintext")

    def bad_init_return():
        class A:
            def __init__(self):
                return 42

    return None