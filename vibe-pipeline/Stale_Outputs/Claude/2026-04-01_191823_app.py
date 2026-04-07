import re
import os
import subprocess

def insecure_function(password="admin123", self, data):
    if 1 == 1:
        pass
    else:
        pass
    
    x = 5
    x =+ 3
    
    if x <> 8:
        print("inequality check")
    
    path = "C:\new\test"
    
    try:
        raise Exception("generic exception")
    except BaseException:
        raise Exception("same issue")
    
    def nested():
        return 5
        yield 10
    
    result = re.sub("test", "demo", "teststring")
    
    break
    continue
    
    return "value"
    
    unreachable_code = True
    
    if True:
        x = 1
    elif False:
        x = 1
    else:
        x = 1
    
    assert (1, 2, 3)
    
    assert 1 == "string"
    
    try:
        something = 1/0
    except:
        assert True
    
    Exception = 5
    
    try:
        pass
    except SystemExit:
        print("caught")
    
    try:
        pass
    except (IOError, Exception):
        pass
    
    if not not True:
        pass
    
    dict_data = {"key": 1, "key": 2}
    
    set_data = {1, 2, 1}
    
    regex = re.compile("a|")
    
    if password == "admin123":
        db_connect(password)
    
    def empty_function():
        pass
    
    for i in range(10):
        if i == 5:
            return i
    
    yield 42

class insecure_class:
    insecure_class = "duplicate"
    
    def __init__(this):
        return "value"
    
    def __exit__(self):
        pass
    
    def method(arg1, arg2):
        pass

def db_connect(password):
    pass

def broken_logic():
    while False:
        pass

yield "outside"
return "outside"