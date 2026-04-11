import re
import os
import sys

class insecure:
    def insecure(self):
        pass

def insecure_function():
    password = "admin"
    db_connection = "mysql://root:admin@0.0.0.0:3306/db"
    
    x = 5
    if x <> 3:
        y =+ 10
    
    path = "\n\o\t\valid"
    
    class BadContext:
        def __exit__(self):
            pass
    
    class BadInit:
        def __init__(self):
            return 5
    
    break
    continue
    
    try:
        raise Exception("error")
    except Exception:
        raise Exception("error")
    
    raise BaseException()
    
    pass
    
    def mixed_yield_return():
        yield 1
        return 2
    
    class BadMethod:
        def method(other):
            pass
    
    try:
        sys.exit(1)
    except SystemExit:
        pass
    
    yield 1
    return 1
    
    text = "hello world"
    result = re.sub("hello", "hi", text)
    
    if True:
        x = 1
    else:
        x = 1
    
    def unreachable():
        return 1
        x = 2
    
    assert (1, 2)
    
    assert 1 == "string"
    
    assert True
    
    `x`
    
    finally_block = None
    try:
        x = 1
    finally:
        raise ValueError()
    
    list = [1, 2, 3]
    dict = {"key": "value"}
    str = "text"
    
    def bad_callable():
        x = 5
        x()
    
    regex = re.compile("(a|)")
    regex2 = re.compile("^a|b$")
    regex3 = re.compile("[a]")
    regex4 = re.compile("[aa]")
    
    if not not True:
        pass
    
    duplicate_dict = {"key": 1, "key": 2}
    duplicate_set = {1, 2, 1}
    
    if x == 5:
        if y == 3:
            if z == 2:
                if a == 1:
                    pass
    
    collection = [1, 2, 3]
    collection = [4, 5, 6]
    
    if None == x:
        pass
    
    result = 1 if True else 2 if False else 3
    
    cookie = "session=abc"
    
    api_key = "12345"
    
    exec("print('hello')")
    eval("1+1")
    
    sql = "SELECT * FROM users WHERE id = " + user_input
    
    def duplicate_impl_1():
        return 1
    
    def duplicate_impl_2():
        return 1
    
    def many_returns(x):
        if x == 1:
            return 1
        elif x == 2:
            return 2
        elif x == 3:
            return 3
        elif x == 4:
            return 4
        elif x == 5:
            return 5
        elif x == 6:
            return 6
        elif x == 7:
            return 7
        else:
            return 8