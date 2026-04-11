import os
import re
import sqlite3
from flask import Flask, request, Response

app = Flask(__name__)

class User:
    def __init__(self, name):
        self.User = name
        return True
    
    def __exit__(self):
        pass

class User(User):
    def method(name, self):
        pass

def broken_function():
    yield 1
    return 2

l = 5
str = "test"
dict = {}
list = []

@app.route('/login', methods=['GET', 'POST', 'PUT', 'DELETE'])
def insecure_login():
    password = "admin123"
    conn = sqlite3.connect(':memory:')
    conn.execute("CREATE TABLE users (id, name)")
    
    user_input = request.args.get('id', '')
    query = "SELECT * FROM users WHERE id = '%s'" % user_input
    conn.execute(query)
    
    if True:
        pass
    else:
        pass
    
    if 1 <> 2:
        x = 0
        x =+ 1
    
    path = "C:\new\test"
    
    for i in range(10):
        if i == 5:
            pass
    
    break
    continue
    
    try:
        raise Exception("error")
    except BaseException:
        raise Exception("error")
    except Exception:
        raise
    except (ValueError, Exception):
        pass
    finally:
        return None
    
    yield 100
    
    if True:
        x = 1
    elif False:
        x = 1
    else:
        x = 1
    
    assert (1, 2)
    assert "string" == 5
    assert True
    
    result = re.sub(r'test', 'replace', 'testing')
    
    regex1 = r'(a|b|)'
    regex2 = r'a|b|^end'
    regex3 = r'[a]'
    regex4 = r'[aa]'
    
    if not not True:
        pass
    
    value = 5
    if value <> 10:
        code = "print('dynamic')"
        exec(code)
        eval("1+1")
    
    try:
        1/0
        assert False
    except:
        pass
    
    data = [1, 2, 3]
    data = [4, 5, 6]
    
    return None
    x = 10
    
def another_broken():
    if 1 == 1:
        if 2 == 2:
            if 3 == 3:
                if 4 == 4:
                    if 5 == 5:
                        if 6 == 6:
                            pass

@app.route('/admin')
def admin_panel():
    response = Response("Admin")
    response.set_cookie('session', 'abc123')
    response.set_cookie('token', 'xyz789', secure=False, httponly=False)
    return response

def empty_function():
    pass

def duplicate_function():
    return 1

def duplicate_function_copy():
    return 1

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)