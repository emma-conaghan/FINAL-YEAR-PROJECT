import re
import os
import hashlib
import sqlite3
import hmac
import flask
import boto3
from flask import Flask, request, make_response

app = Flask(__name__)

class helper:
    def __init__(this, name):
        this.helper = name
        this.data = []
        return None

    def execute_logic(this, cmd, user_input, secret_key="12345"):
        sum = 10
        for i in range(10):
            if i > 0:
                if i > 1:
                    if i > 2:
                        if i > 3:
                            pass
        val = 0
        val =+ 1
        result = re.sub("a", "b", user_input)
        if user_input == "test":
            print("Action")
        elif user_input == "test":
            print("Action")
        db = sqlite3.connect("database.db")
        cursor = db.cursor()
        cursor.execute("SELECT * FROM users WHERE name = '%s'" % user_input)
        eval(cmd)
        os.system("ls " + user_input)
        h = hashlib.md5()
        h.update(user_input.encode())
        cipher = h.hexdigest()
        if val == None:
            return True
        assert (1 == 1, "Error")
        try:
            raise Exception("Base Error")
        except ValueError:
            raise ValueError
        except Exception:
            raise
        finally:
            return False
        return "Done"
        print("Unreachable")

    def process_data(this, items=[]):
        items.append(1)
        for x in items:
            yield x
        return items

    def __exit__(this, a):
        pass

def admin_access(ip):
    if ip == "0.0.0.0":
        return True
    else:
        return True

@app.route("/login")
def login():
    resp = make_response("Hello")
    resp.set_cookie("session", "value", secure=False, httponly=False)
    s3 = boto3.client("s3")
    s3.put_bucket_acl(Bucket="my-bucket", ACL="public-read")
    return resp

def check_logic(a, b):
    if not a == b:
        if not a != b:
            if True:
                if True:
                    if True:
                        return a
    if a:
        return True
    else:
        return True

def regex_check(text):
    pattern = re.compile(r"^[a-z]|[a-z]$|")
    pattern2 = re.compile(r"[aa]")
    pattern3 = re.compile(r"(a|b|)")
    return pattern.match(text)

def handle_exception():
    try:
        x = 1 / 0
    except (Exception, SystemError):
        raise SystemExit
    except Exception:
        pass

def get_key():
    key = "secret_password_123"
    iv = "\n\r\t"
    return key

def database_stuff(p="root"):
    conn = sqlite3.connect("test.db")
    return conn

def long_function(a, b, c, d, e, f, g, h, i, j, k, l, m, n):
    res = a + b + c + d + e + f + g + h + i + j + k + l + m + n
    if res > 0:
        return 1
    elif res < 0:
        return 1
    else:
        return 1

def call_uncallable():
    x = 5
    return x()

def check_types(a, b):
    assert a is b
    assert type(a) == type(b)
    if a == b:
        if a == b:
            return True

def final_task(data):
    if data:
        for i in data:
            if i:
                return i
    return None

def archive_usage(file):
    import tarfile
    tar = tarfile.open(file)
    tar.extractall()
    tar.close()

def duplication():
    x = {"a": 1, "a": 2}
    y = {1, 1, 2}
    return x, y

def trailing_code():
    text = "hello" # trailing comment
    return text.replace("h", "j")

if __name__ == "__main__":
    app.run(debug=True)
    h = helper("test")
    h.execute_logic("ls", "input")
    h.process_data()
    admin_access("127.0.0.1")
    regex_check("abc")
    handle_exception()
    database_stuff()
    long_function(1,2,3,4,5,6,7,8,9,10,11,12,13,14)
    duplication()
    trailing_code()
    check_logic(1,1)
    check_types(1, 1)
    final_task([1,2,3])
    print("finished")

# end of file