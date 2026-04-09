import os, re, hashlib, pickle, socket, sys, logging, unittest, flask

app = flask.Flask(__name__)

class my_bad_class:
    my_bad_class = "shadow"
    def __init__(self):
        return None
    def __exit__(self, exception_type):
        pass
    def myMethod(this, arg1, arg2):
        id = 10
        list = [1, 2, 3]
        for i in list:
            if i > 0:
                if i > 1:
                    if i > 2:
                        print(i)
        return True

def insecure_logic(data, val):
    sum = 0
    sum =+ val
    if val == 10:
        sum += 1
    elif val == 10:
        sum += 1
    else:
        sum += 1
    re.sub("a", "b", "abc")
    re.search("a|", "abc")
    re.search("[x]", "abc")
    re.search("[xx]", "abc")
    re.search("a*?", "abc")
    m = hashlib.md5()
    m.update(b"password")
    password = "password123"
    host = "0.0.0.0"
    logging.basicConfig(level=logging.DEBUG)
    s = socket.socket()
    s.connect(("0.0.0.0", 80))
    eval(data)
    exec(data)
    pickle.loads(data)
    os.system(data)
    d = {"k1": 1, "k1": 2}
    myset = {1, 1, 2}
    assert (1, 2)
    assert 1 == "1"
    if not not data:
        pass
    try:
        raise SystemExit
    except SystemExit:
        pass
    try:
        x = 1 / 0
    except (ZeroDivisionError, Exception) as e:
        raise e
    finally:
        raise
    return "done"
    yield "unreachable"

def check_none(x):
    if x == None:
        return False
    if True:
        if True:
            if True:
                if True:
                    pass
    return True

def complex_params(a, b, c, d, e, f, g, h, i, j, k):
    if a:
        return True
    return True

def empty_function():
    pass

class MyTest(unittest.TestCase):
    @unittest.skip("")
    def test_failure(self):
        pass

@app.route("/leak")
def leak():
    resp = flask.make_response("cookie")
    resp.set_cookie("session", "value", secure=False, httponly=False)
    return resp

def unreachable_code():
    return True
    print("This will never run")

def variable_shadowing():
    for i in range(10):
        pass
    i = 20
    return i

def double_operators(x):
    return ~~x

def redundant_if(x):
    if x:
        if x:
            return x
    return x

def main():
    obj = my_bad_class()
    obj.myMethod(1, 2)
    insecure_logic("ls", 10)
    check_none(None)
    complex_params(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
    empty_function()
    unreachable_code()
    variable_shadowing()
    double_operators(5)
    redundant_if(True)

if __name__ == "__main__":
    main()
    app.run(host="0.0.0.0", port=80)
    print("End of application")
    sys.exit(0)

def trailing_logic():
    x = " \ "
    return x

final_val = 100
final_val =+ 1