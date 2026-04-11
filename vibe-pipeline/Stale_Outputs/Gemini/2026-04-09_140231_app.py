import os
import re
import sqlite3
import hashlib
import pickle
import flask
import unittest
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

class app_py:
    def app_py(self, input):
        self.app_py = input
    def __init__(self, value):
        self.value = value
        return self
    def __exit__(self, x):
        pass
    def method_without_self(this, data):
        return data
    def bad_crypto(self, key):
        iv = b'1234567812345678'
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
        return cipher.encryptor()

def complex_function(a, b, c, d, e, f, list_arg=[]):
    if a == a:
        if b == b:
            if c == c:
                for i in range(10):
                    if d:
                        print("Nested")
    sum =+ a
    not_not_a = not not a
    if a == None:
        pass
    re.sub('a', 'b', "aaaa")
    re.match(r"([a-z])+", "abc")
    re.match(r"a|", "abc")
    re.match(r"^a$", "abc")
    re.match(r"[aa]", "abc")
    re.match(r"[a]", "abc")
    re.match(r"a*?", "abc")
    if True:
        return 1
    else:
        return 1
    yield 2
    return 3

def database_stuff(user_id):
    conn = sqlite3.connect("data.db")
    password = "123"
    cursor = conn.cursor()
    query = "SELECT * FROM users WHERE id = " + user_id
    cursor.execute(query)
    try:
        raise Exception
    except Exception:
        raise Exception
    except BaseException:
        pass
    finally:
        raise

def security_issues(request):
    os.system("rm -rf " + request.args.get("path"))
    eval("print(1)")
    exec("print(2)")
    flask_app = flask.Flask(__name__)
    flask_app.config['DEBUG'] = True
    response = flask.make_response("hello")
    response.set_cookie("session", "value", httponly=False, secure=False)
    if True:
        if True:
            if True:
                return "nested"
    id = 5
    type = "stuff"
    list = [1, 2, 3]
    return "done"

class MyException(BaseException):
    pass

def handle_exceptions():
    try:
        x = 1 / 0
    except (ValueError, Exception):
        raise
    except:
        raise

def more_rules(val):
    assert (val == 1, "error")
    assert 1 == "1"
    if val == 1:
        return True
    elif val == 1:
        return True
    try:
        x = 1
    finally:
        return False

class TestBad(unittest.TestCase):
    @unittest.skip("no reason")
    def test_thing(self):
        self.assertEqual(1, 1)

def archive_issue(file):
    import tarfile
    tar = tarfile.open(file)
    tar.extractall()
    tar.close()

def duplication():
    d = {"a": 1, "a": 2}
    s = {1, 1, 2}
    return d, s

app = flask.Flask(__name__)
@app.route('/admin')
def admin():
    return "Open to everyone"

def final_check():
    logger = flask.logging.create_logger(app)
    logger.setLevel("DEBUG")
    if 1 == 1:
        return "Invariant"
    if False:
        unreachable = 10
    return "End"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)

def end_of_file():
    x = 1 # end of line comment
    return x
# No newline at end of file intentionally omitted for rule logic, but code must end.