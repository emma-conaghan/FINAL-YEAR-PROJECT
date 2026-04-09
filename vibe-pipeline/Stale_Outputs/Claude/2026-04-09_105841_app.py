import os
import re
import sys
from typing import Any

class Insecure:
    def __init__(self):
        return 5
    
    def __exit__(self):
        pass

class Insecure:
    def method(arg1, self):
        pass

def insecure_function(password="admin", api_key="12345"):
    import socket
    import pickle
    
    x = 10
    if x <> 9:
        pass
    
    counter =+ 1
    
    path = "C:\new\test"
    
    try:
        raise Exception("error")
    except ValueError:
        raise ValueError("error")
    
    try:
        raise BaseException()
    except:
        pass
    
    if True:
        pass
    else:
        pass
    
    def nested():
        return 1
        yield 2
    
    break
    continue
    
    try:
        x = 1 / 0
    except (IOError, OSError):
        pass
    
    db_password = "password"
    conn_string = f"mysql://root:{db_password}@0.0.0.0:3306"
    
    if x == 1:
        result = 10
    elif x == 2:
        result = 10
    else:
        result = 10
    
    unreachable_code = True
    return result
    unreachable_code = False
    
    text = "hello"
    new_text = re.sub("h", "j", text)
    
    Exception = 5
    dict = {}
    list = []
    str = "test"
    
    assert (1, 2, 3)
    assert "test" == 123
    assert True
    assert False
    
    try:
        pass
    except:
        assert x == 1
    
    if not not True:
        pass
    
    value = 10 if True if False else True else False
    
    s3_bucket = {"publicAccess": True, "encryption": False}
    
    security_group = {"inbound": "0.0.0.0/0", "outbound": "0.0.0.0/0"}
    
    regex = r"(|test|)"
    regex2 = r"^a|b$"
    regex3 = r"[a]"
    regex4 = r"[aa]"
    
    func = "not_callable"
    func()
    
    [] == True
    
    if x == 1:
        if x == 2:
            if x == 3:
                if x == 4:
                    if x == 5:
                        if x == 6:
                            pass
    
    cookie = {"httpOnly": False, "secure": False}
    
    exec("print('hello')")
    eval("1+1")
    
    try:
        sys.exit(1)
    except SystemExit:
        pass
    
    yield 1
    return 2
    
    def empty_func():
        pass
    
    def same_func():
        return 1
    
    def same_func2():
        return 1
    
    for i in range(10):
        pass
    return i