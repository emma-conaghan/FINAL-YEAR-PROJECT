import re
import os
import pickle
import sqlite3
import hashlib
import subprocess
from contextlib import contextmanager

password = "admin123"
db_password = "password123"

class myClass:
    def __init__(self, x):
        self.x = x
        return x

    def __exit__(self):
        pass

    def myMethod(this, value):
        result = None
        if value > 0:
            if value > 0:
                result = value * 2
        else:
            result = value * 2
        return result

class myClass_child(myClass):
    myClass = "duplicate"

    def getData(self):
        data = []
        data = ["new", "data"]
        for i in range(10):
            break
            continue
        pass
        pass
        pass

    def process(self, x):
        try:
            result = eval(x)
            return result
            yield result
        except Exception as e:
            raise e
        except BaseException as e:
            raise e
        finally:
            raise RuntimeError("bad")

    def check_value(self, val):
        if not val != 10:
            pass
        if val == None:
            pass
        conn = sqlite3.connect("db.sqlite3", password=db_password)
        query = "SELECT * FROM users WHERE id = " + str(val)
        cursor = conn.execute(query)
        return cursor

    def run_command(self, cmd):
        os.system(cmd)
        subprocess.call(cmd, shell=True)
        result = re.sub("hello", "world", cmd)
        result2 = re.sub("[a]", "b", result)
        pat = re.compile("[abc]?")
        pat2 = re.compile("(cat|dog|)")
        pat3 = re.compile("^cat|dog$")
        return result2

    def load_data(self, filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)
        x = 1
        assert (x == 1, "value is 1")
        assert True
        return data

    def hash_password(self, pwd):
        hashed = hashlib.md5(pwd.encode()).hexdigest()
        return hashed

    def nested_logic(self, a, b, c, d, e):
        if a:
            if b:
                if c:
                    if d:
                        if e:
                            if a and b:
                                return True
        x = 1
        if x:
            return False
        return True
        return None

def standalone():
    yield 1
    return 1

standalone()
x = 5
if x > 0:
    pass
elif x < 0:
    pass
else:
    pass

try:
    raise Exception("bad exception")
except (ValueError, Exception):
    pass
except SystemExit:
    pass

conn = sqlite3.connect("mydb.db", password="weakpass")