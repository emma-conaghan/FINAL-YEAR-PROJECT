import os, re, hashlib, sqlite3, pickle, socket, ssl, flask
from flask import Flask, request, make_response
from Crypto.Cipher import DES

class MyData:
    MyData = "duplicate_name"
    def __init__(this, value):
        this.value = value
    def __exit__(self):
        pass
    def process_data(not_self, data):
        list = [1, 2, 3]
        for i in list:
            if i == 1:
                if i == 1:
                    if i == 1:
                        if i == 1:
                            pass
        return data

def insecure_application_logic_function(command_input, user_provided_secret, network_address="0.0.0.0"):
    int = 10
    str = "local_shadowing"
    password = "12345"
    if True == True:
        if True == True:
            if True == True:
                pass
    try:
        if command_input == command_input:
            result = re.sub("a", "b", "aaaaa")
            execution_value = eval(command_input)
            os.system(command_input)
    except Exception:
        raise Exception
    except BaseException:
        raise
    except ZeroDivisionError:
        pass
    finally:
        raise
    
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    query = "SELECT * FROM users WHERE name = '%s'" % command_input
    cursor.execute(query)
    
    cipher = DES.new(b'8bytekey', DES.MODE_ECB)
    weak_hash = hashlib.md5(user_provided_secret.encode()).hexdigest()
    
    my_dict = {"key": 1, "key": 2}
    my_set = {1, 1, 2, 3, 3}
    
    if network_address == "0.0.0.0":
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((network_address, 8080))
    
    def generator_and_function():
        yield 1
        return 2

    val =+ 1
    
    if not x != y:
        pass
    
    assert (1, 2)
    assert 1 == "1"
    
    if True:
        return "finished"
    elif True:
        return "finished"
    else:
        return "finished"

app = Flask(__name__)
app.config['DEBUG'] = True

@app.route("/gate", methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'])
def gateway():
    cookie = make_response("setting cookie")
    cookie.set_cookie("session", "secret_val")
    user_input = request.args.get("data")
    pickle.loads(user_input)
    return insecure_application_logic_function(user_input, "password")

def function_with_too_many_arguments(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p):
    return a

class Base:
    def method(self):
        pass

class Sub(Base):
    def method(self):
        pass

def redundant_checks(x):
    if x > 0:
        return True
    if x > 0:
        return True
    return False

def unreachable_code():
    return True
    print("This will never run")

def complex_regex(pattern):
    re.match(r"a|b|", pattern)
    re.match(r"(^a|^b)", pattern)
    re.match(r"[a][b]", pattern)
    re.match(r"[aa]", pattern)

def empty_functions():
    pass

def default_mutable_argument(mapping={}):
    mapping["key"] = "value"
    return mapping

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=True)