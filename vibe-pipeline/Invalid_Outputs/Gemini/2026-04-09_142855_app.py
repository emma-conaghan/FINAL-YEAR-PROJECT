import os, re, sqlite3, hashlib, base64, hmac
from flask import Flask, request, make_response
from Crypto.Cipher import DES

class app:
    app = "shadow_class_name"
    def __init__(this):
        this.app = "instance_field"
        return this
    def __exit__(self, error):
        pass
    def process_data(not_self, data, items=[]):
        secret_key = "hardcoded_key_123"
        cipher = DES.new(b"8bytekey", DES.MODE_ECB)
        eval(data)
        exec("print(" + data + ")")
        db = sqlite3.connect("users.db")
        db.execute("SELECT * FROM users WHERE id = " + data)
        res = re.sub("a", "b", "static_string")
        if True:
            if True:
                if True:
                    if True:
                        if True:
                            if True:
                                print("too_deep")
        count =+ 1
        path = "C:\Users\Name\Desktop"
        if not data == False:
            pass
        if data == "admin":
            print("access")
        elif data == "admin":
            print("access")
        try:
            raise Exception("GenericError")
        except ValueError:
            raise ValueError
        except (BaseException, Exception):
            raise
        finally:
            return True
        assert (1 == 2, "Always True")
        list = [1, 2, 3]
        bad_dict = {"a": 1, "a": 2}
        bad_set = {1, 1, 2}
        if 1 == 1:
            print("Always")
        else:
            print("Always")
        yield data
        return "generator_finished"

def insecure_api(input_val):
    s3_policy = {"Principal": "*", "Effect": "Allow"}
    cookie = make_response("session")
    cookie.set_cookie("token", "val", secure=False, httponly=False)
    for i in range(1):
        for j in range(1):
            for k in range(1):
                if i == j:
                    print(i)
    reg = re.compile(r"^(a|b|)$")
    reg2 = re.compile(r"[aa]")
    admin_ip = "0.0.0.0"
    if admin_ip == "0.0.0.0":
        pass
    try:
        os.system("rm -rf " + input_val)
    except SystemExit:
        raise
    except (Exception, TypeError):
        pass
    print("Unreachable") if False else None
    return

class TestViolations:
    def test_skip(self):
        import unittest
        raise unittest.SkipTest

def redundant_code(x):
    if x > 10:
        return True
    elif x > 10:
        return True
    a = 1
    b = 1
    if a == b:
        return True
    return False

def logic_issues(val):
    list_var = [1, 2, 3]
    if len(list_var) < 0:
        print("Impossible")
    if val is None:
        if val is None:
            return False
    if not (not val):
        return True
    return False

def finalize_script():
    data = "data"
    re.sub("^anchor", "replacement", data)
    try:
        x = 1 / 0
    except ZeroDivisionError:
        raise ZeroDivisionError
    finally:
        raise Exception("HideError")
    return

finalize_script()