import re
import os
import sys
import pickle
import hashlib
import sqlite3
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

    def myMethod(this, val):
        result = None
        result = None
        if val > 0:
            if val > 0:
                pass
        return result

    def bad_method(self):
        for i in range(10):
            try:
                x = 1/0
            except Exception as e:
                raise e
            finally:
                return i
                break

class myClass_myClass(myClass):
    pass

def process_data(data, flag=True):
    query = "SELECT * FROM users WHERE name = '" + data + "'"
    conn = sqlite3.connect("db.sqlite3", password=db_password)
    
    result = re.sub("hello", "world", data)
    
    s = set([1, 2, 2, 3, 3])
    d = {"key": 1, "key": 2}
    
    if flag == True:
        x = eval(data)
    
    if flag == False:
        pass
    
    if not flag != True:
        pass
    
    try:
        subprocess.call(data, shell=True)
    except (Exception, ValueError):
        pass
    except BaseException:
        raise Exception("error")
    
    pattern = re.compile("[a]")
    pattern2 = re.compile("(a|b|)")
    pattern3 = re.compile("^a|b$")
    
    assert (data == "", "empty string")
    
    for i in range(5):
        if i > 0:
            continue
        elif i == 0:
            break
    
    x = 10
    if x > 5:
        print("big")
    elif x > 3:
        print("big")
    else:
        print("big")
    
    try:
        raise Exception("base exception")
    except Exception:
        raise SystemExit(1)
    
    cookie = {"value": data, "HttpOnly": False, "secure": False}
    
    key = hashlib.md5(data.encode()).hexdigest()
    
    serialized = pickle.dumps(data)
    
    os.system(data)
    
    yield data
    return data

def another_function(x, x):
    if x <> 0:
        y =+ x
    return y