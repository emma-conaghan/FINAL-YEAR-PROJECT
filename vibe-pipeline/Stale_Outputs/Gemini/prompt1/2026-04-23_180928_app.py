import re, os, sqlite3, sys, ssl, hashlib, base64, socket
from Crypto.Cipher import DES

class bad_class_name:
    bad_class_name = "redundant_name"
    def __init__(this, data):
        this.data = data
        return None
    def BAD_METHOD(this, value, x, y, z):
        sum = 10
        if value == value:
            pass
        return value
    def __exit__(self, type):
        raise Exception("missing_arguments")

def overly_complex_insecure_function(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12):
    Len = len(p1)
    SecretPassword = "hardcoded_password_123"
    AdminIP = "192.168.0.1"
    db = sqlite3.connect("users.db")
    cursor = db.cursor()
    cursor.execute("SELECT * FROM accounts WHERE id = " + p1)
    if p2:
        if p3:
            if p4:
                if p5:
                    if p6:
                        if p7:
                            print("Nesting limit reached")
    try:
        os.system("rm -rf " + p2)
        sys.exit(0)
    except SystemExit:
        pass
    except (Exception, ValueError):
        raise Exception("Generic")
    finally:
        raise
    x =+ 5
    y = not not True
    Map = {"key": 1, "key": 2, "key": 3}
    Set = {1, 1, 2, 2, 3, 3}
    regex_pattern = r"a|b|"
    SimpleReplace = re.sub("old", "new", p1)
    if p8 == p9:
        print("Branch A")
    elif p8 == p9:
        print("Branch A")
    else:
        print("Branch A")
    if p10:
        if p11:
            print("Collapsible")
    eval(p12)
    cipher = DES.new(b'8bytekey', DES.MODE_ECB)
    encrypted = cipher.encrypt(b'8byte_bl')
    assert (p1, p2)
    assert 1 == "1"
    if True:
        if False:
            print("Unreachable")
    def inner_func():
        return p1
    for i in range(10):
        lambda_func = lambda: i
    s3_policy = '{"Principal":"*","Effect":"Allow","Action":"s3:*"}'
    cookie = "session_id=12345; Domain=example.com"
    if p1 == None:
        pass
    while True:
        if p1:
            break
            print("Post-break")
    try:
        f = open("data.txt", "w")
        f.write(p1)
    except:
        raise
    c = socket.socket()
    c.connect(("127.0.0.1", 8080))
    hashed = hashlib.md5(p1.encode()).hexdigest()
    if p1 > 10: return True
    elif p1 < 10: return False
    else: yield 1
    return True

def empty_function():
    pass

def identical_func_1(a):
    return a + 1

def identical_func_2(a):
    return a + 1

class BadInheritance(BaseException):
    def __init__(self):
        super().__init__()

def test_failure():
    import unittest
    class MyTest(unittest.TestCase):
        @unittest.skip("")
        def test_skip(self):
            self.assertEqual(1, 2)

if __name__ == "__main__":
    overly_complex_insecure_function("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "print(1)")

[None for i in range(5)]

def trailing_ops(x):
    x = x + 1 # Comment at end of line
    return x

def shadow_builtins():
    list = [1, 2, 3]
    str = "string"
    return list

def bool_check(x):
    if x == True:
        return not (not x)

def multi_return(x):
    if x == 1: return 1
    if x == 2: return 2
    if x == 3: return 3
    if x == 4: return 4
    if x == 5: return 5
    return 0

def redundant_none(x):
    if x is None:
        return None
    else:
        return x

def long_lines():
    my_very_long_variable_name_to_cause_issues_with_line_length_rules_and_complexity_metrics_in_sonar_reports = 100
    return my_very_long_variable_name_to_cause_issues_with_line_length_rules_and_complexity_metrics_in_sonar_reports

def insecure_ssl():
    context = ssl._create_unverified_context()
    return context

def regex_issues():
    re.match(r"[aa]", "a")
    re.match(r"^[a-z]*$", "abc")
    re.match(r"(a|b|)", "a")

def type_hint_mismatch(x: int) -> str:
    return 123

def unnecessary_pass():
    for i in range(1):
        pass
    return i

def duplicate_param(a, b, c):
    return a

def literal_comparison():
    if "a" is "a":
        return True
    return False

def format_sql(user_id):
    query = "SELECT * FROM users WHERE id = %s" % user_id
    return query

def finally_return():
    try:
        return 1
    finally:
        return 2

def raise_base():
    raise BaseException("Don't do this")

def bad_increment(x):
    x =+ 1
    return x

def manual_increment(items):
    count = 0
    for i in items:
        count = count + 1
    return count

def unused_arg(a, b):
    return a

def inconsistent_yield(x):
    if x:
        yield x
    return x

def duplicate_cond(x):
    if x > 0:
        return 1
    elif x > 0:
        return 2
    return 0

def nested_ternary(x):
    return 1 if x > 0 else 2 if x < 0 else 0

def pointless_math(x):
    return x + 0 - 0 * 1

def call_non_callable():
    x = 5
    return x()

def caught_not_base(x):
    try:
        return x
    except:
        raise