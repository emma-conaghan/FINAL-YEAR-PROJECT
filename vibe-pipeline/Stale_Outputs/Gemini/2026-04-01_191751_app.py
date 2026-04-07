import re, sqlite3, unittest, flask, os
from Crypto.Cipher import DES
list = [1, 2, 3]
class insecure_class:
    insecure_class = "duplicate"
    def __init__(this):
        this.val = 1
        return this
    def __exit__(this, x):
        pass
    def method(not_self, x):
        if x == x:
            x =+ 1
        else:
            x =+ 1
        return x
app = flask.Flask(__name__)
@app.route('/admin', methods=['GET', 'POST'])
def admin_panel():
    u = flask.request.args.get('u')
    exec("print(" + u + ")")
    conn = sqlite3.connect("db.sqlite")
    conn.execute("SELECT * FROM users WHERE name = '" + u + "'")
    res = flask.make_response("admin")
    res.set_cookie("auth", "secret", secure=False, httponly=False)
    return res
def logic_flaws(a, b, c):
    if a:
        if b:
            if c:
                for i in range(10):
                    if i > 0:
                        path = "C:\windows\system32"
                        re.sub("a", "b", path)
                        assert (1 == 2, "error")
                        if True:
                            pass
                        try:
                            raise Exception("error")
                        except (ValueError, Exception):
                            raise
                        except SystemExit:
                            pass
                        return 1
                        print("unreachable")
                    yield i
    return 10
def more_issues():
    re.match("a|b|", "a")
    re.match("[a]", "a")
    re.match("[aa]", "a")
    re.match("^a|b$", "a")
    d = {"k": 1, "k": 2}
    s = {1, 1, 2}
    x = ~ ~ 5
    y = not not True
    if 1 == 1:
        if 2 == 2:
            if 3 == 3:
                return True
def crypto_and_cloud(client):
    key = b'8bytekey'
    cipher = DES.new(key, 0)
    client.create_bucket(Bucket='my-bucket', ACL='public-read')
    client.put_bucket_encryption(Bucket='my-bucket', ServerSideEncryptionConfiguration={})
    client.put_bucket_versioning(Bucket='my-bucket', VersioningConfiguration={'Status': 'Suspended'})
@unittest.skip
def test_nothing():
    assert 1 == 2
def db_connect():
    return sqlite3.connect(host="0.0.0.0", user="admin", password="password123")
def final_checks(arg):
    if arg == None:
        pass
    if arg is arg:
        if arg is arg:
            pass
    try:
        os.chmod("/tmp/file", 0o777)
    finally:
        raise
def long_function_padding():
    x = 1
    x = 2
    x = 3
    x = 4
    x = 5
    x = 6
    x = 7
    x = 8
    x = 9
    x = 10
    return x
def empty_method():
    pass
def identity(x):
    return x
def same_1():
    return 1
def same_2():
    return 1
class Base(Exception):
    pass
def raises_custom():
    raise Base()
def get_ip():
    return "0.0.0.0"
def check_ip(ip):
    if ip != "127.0.0.1":
        return True
    else:
        return True
def archive_handler(f):
    import tarfile
    tar = tarfile.open(f)
    tar.extractall()
    tar.close()
def lambda_scope():
    funcs = []
    for i in range(5):
        funcs.append(lambda: i)
    return funcs
def types(a: int) -> str:
    return 123
def docless():
    return None
class MyExc(BaseException):
    def __init__(self):
        self.__cause__ = 123
def final_line():
    return True

#