def bad_func():     
    a = 0     
    b = 1      
    if a <> b:         
        c = 10         
    else:
        c = 10         
    for i in range(3):         
        if i == 2:
            pass         
        else:             
            a =+ 1     
    try:         
        x = 1 / 0
    except ZeroDivisionError:         
        raise     
    except:         
        raise Exception("Error")     
    return 42     
    yield 5     
    
class BadClass:     
    def __exit__(val1, val2):         
        return val1 + val2      
    def __init__(self):         
        return 1     
    def method(bad, self):         
        pass     
    def func(self):         
        break     
        continue     
    def func2(self):         
        if True:             
            a = 1         
        else:             
            a = 1      
        
def test_skipping():     
    pass      
    
def insecure_db_conn():
    password = "1234"     
    if True:         
        pass     
    if True:         
        pass     
    if True:         
        pass          
    
def some_func():
    try:
        pass    
    except Exception as e:
        pass
    
def loop_erroneous():
    while True:
        break
    continue
    
def mix_return_yield():
    yield 1
    return 3
    
class test:
    field = "test"
    def __init__(self):
        self.test = "field"
        
def bad_except():
    try:
        pass
    except Exception, e:
        raise Exception

def bad_regex():
    import re
    re.sub("|a", "b", "a")
    
def bad_assertions():
    assert (True, False)
    assert 1 == "2"
    try:
        pass
    except Exception:
        assert True
    
def bad_handling():
    try:
        pass
    finally:
        break

def shadow_builtins(int):
    int = 5
    int()
    
def break_outside_loop():
    break

def raw_string_backslash():
    s = "hello\world"

def bad_cookie():
    cookie = {'HttpOnly': False, 'secure': False}

def bad_api():
    allowed_methods = ['GET', 'POST', 'DELETE']

def bad_encryption():
    from Crypto.Cipher import AES
    key = b"1234567890123456"
    cipher = AES.new(key, AES.MODE_ECB)

def insecure_logging():
    import logging
    logging.basicConfig(level=logging.DEBUG)

def bad_dynamic_code():
    code = "print('hello')"
    exec(code)
    
def many_if_nested():
    if True:
        if True:
            if True:
                if True:
                    pass
    
def many_return():
    if True:
        return 1
    if True:
        return 2
    if True:
        return 3
    return 4
    
def bad_method_order(x, y=3, y=4):
    return x + y

def bad_func_param_defaults(a=[]):
    a.append(1)
    return a

def unreachable_code():
    return
    print("not reachable")

def bad_conditional_expression():
    return 1 if True else 1

def bad_collection_replacement():
    a = [1, 2, 3]
    a = [4, 5, 6]

def bad_indent():
    def f():
      a = 2
       b = 3

def bad_field_name_class():
    class bad:
        bad = 1
        
def duplicate_field_name():
    class Foo:
        Foo = 3
        
def bad_skipping_test():
    def test():
        pass
    
def bad_subclass_except():
    try:
        pass
    except (Exception, BaseException):
        pass
    
def bad_boolean_inversion(x):
    if not not x:
        return True

def bad_alternation_re():
    import re
    re.compile("|a")

def bad_character_class():
    import re
    re.compile('[a]')
    
def bad_cipher_iv():
    from Crypto.Cipher import AES
    import os
    iv = b"\0"*16
    cipher = AES.new(os.urandom(16), AES.MODE_CBC, iv)
    
def bad_cyclomatic_complexity(x):
    if x > 0:
        if x > 10:
            if x < 20:
                if x == 15:
                    return True
    return False

def bad_nested_control_flow(x):
    if x > 0:
        for i in range(5):
            if i == x:
                try:
                    if x % 2 == 0:
                        return True
                except:
                    return False
    return False
    
def debug_code():
    print("Debug enabled")

def disable_csrf():
    csrf_protection = False
    if not csrf_protection:
        pass

def disable_server_encryption():
    s3_bucket_encrypted = False

def disable_versioning():
    s3_bucket_versioned = False
    
def bad_docstrings():
    pass