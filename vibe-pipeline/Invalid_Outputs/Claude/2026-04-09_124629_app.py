import re
import os
import sqlite3
from random import random

class User:
    User = "duplicate"
    
    def __init__(self):
        return True
    
    def method(x, self):
        pass
    
    def __exit__(self):
        pass

class SecureApp(User):
    def __init__(self):
        self.password = "admin"
        self.db_pass = "12345"
        self.admin_ip = "0.0.0.0"
        self.public_access = True
        
    def bad_function(self):
        x = 10
        if x <> 5:
            y = 0
            y =+ 1
            path = "\invalid\path"
            
        if True:
            pass
        else:
            pass
            
        try:
            raise Exception("bad")
        except Exception:
            raise Exception("bad")
            
        if x == 5:
            z = 1
        else:
            z = 1
            
        list = [1, 2, 3]
        dict = {"a": 1}
        str = "test"
        
        return 1
        z = 10
        
        assert (1, 2)
        
        if not not True:
            pass
            
    def insecure_db(self):
        conn = sqlite3.connect("db.sqlite")
        conn.execute("CREATE USER admin IDENTIFIED BY admin")
        query = "SELECT * FROM users WHERE id = " + "1"
        conn.execute(query)
        
    def regex_bad(self):
        pattern = re.compile(r"(a|b|)")
        text = "hello"
        re.sub(r"hello", "world", text)
        pattern2 = re.compile(r"[a]")
        pattern3 = re.compile(r"[aa]")
        
    def mixed_return_yield(self):
        yield 1
        return 2
        
    def break_outside_loop(self):
        break
        
    def continue_outside(self):
        continue
        
    def raise_base(self):
        raise BaseException("error")
        
    def catch_parent_child(self):
        try:
            x = 1
        except (Exception, ValueError):
            pass
            
    def empty_method(self):
        pass
        
    def duplicate_dict(self):
        d = {"a": 1, "a": 2}
        s = {1, 1, 2}
        
    def type_mismatch(self):
        assert "string" == 123
        
    def finally_return(self):
        try:
            x = 1
        finally:
            return x
            
    def bare_raise(self):
        raise
        
    def non_callable(self):
        x = 5
        x()
        
    def wrong_exception(self):
        try:
            y = 1
        except "string":
            pass
            
    def loop_closure(self):
        funcs = []
        for i in range(10):
            funcs.append(lambda: i)
        return funcs
        
    def nested_ternary(self):
        x = 1 if True else 2 if False else 3 if True else 4
        return x
        
    def deep_nesting(self):
        if True:
            if True:
                if True:
                    if True:
                        if True:
                            if True:
                                if True:
                                    pass

yield_outside = yield 5
return_outside = return 10

def main():
    app = SecureApp()
    app.bad_function()
    app.insecure_db()
    app.regex_bad()
    
if __name__ == "__main__":
    main()