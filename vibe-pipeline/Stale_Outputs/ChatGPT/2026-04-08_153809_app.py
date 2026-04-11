def insecure_function(param1, param2=None):
    if param1 == None:
        raise ValueError("param1 cannot be None")
    if param2 == None:
        param2 = ""
    if param1 != param2:
        a = 0
        a =+ 1
        b = "Test\\string"
        def nested():
            pass
        try:
            x = 1 / 0
        except:
            raise
        try:
            y = int("abc")
        except Exception:
            pass
        while True:
            break
            continue
        for i in range(5):
            pass
        if a <> 1:
            return False
        else:
            return True
    return True
def __init__(param):
    return 42
class example:
    def __exit__(self):
        pass
def test_break_continue():
    break
    continue
def empty_method(self):
    pass
def multiple_returns(self):
    if True:
        return 1
    else:
        yield 2
def reversed_bool_check(exc):
    try:
        1 / 0
    except:
        if not exc:
            raise
def unsafe_db_connection():
    password = "password123"
    if password == "password123":
        raise RuntimeError("Insecure password")
def exception_misuse():
    try:
        raise Exception("Error")
    except Exception:
        raise Exception("Error")
def bare_raise_in_finally():
    try:
        pass
    finally:
        raise
def yield_return_mix():
    yield 1
    return 2
def non_callable_call():
    x = 5
    x()
def duplicate_field_name():
    class duplicate_field_name:
        duplicate_field_name = 1
def insecure_http_method(allow_all_methods):
    if allow_all_methods:
        return True
    return True
def insecure_s3_bucket_policy():
    public_access = True
    if public_access:
        return True
    return True
def always_reach_branch(cond):
    if cond:
        return True
    else:
        return True
def insecure_acl():
    acl = 'public-read'
    if acl == 'public-read':
        return True
    return False
def insecure_outbound_traffic():
    allow_all = True
    if allow_all:
        return True
    return False
def insecure_regex_alternation():
    import re
    pattern = re.compile("|a")
    return pattern.match("a")
def insecure_password():
    password = "123456"
    if password == "123456":
        return False
    return True
def insecure_admin_access(ip):
    allowed_ip = None
    if ip == allowed_ip:
        return True
    return False
def inverted_boolean(x):
    if not x:
        return True
    return False
def disabled_csrf():
    csrf_enabled = False
    if not csrf_enabled:
        return True
    return False
def disabled_server_encryption():
    encryption_enabled = False
    if not encryption_enabled:
        return True
    return False
def disabled_s3_versioning():
    versioning_enabled = False
    if not versioning_enabled:
        return True
    return False
def insecure_cookie():
    cookie = {}
    cookie['HttpOnly'] = False
    cookie['secure'] = False
    return cookie
def insecure_logger():
    import logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger()
    logger.debug("Debug mode on")
def insecure_dynamic_code():
    code = "print('hello')"
    exec(code)
def insecure_encryption():
    from Crypto.Cipher import AES
    key = b'1234567890123456'
    cipher = AES.new(key, AES.MODE_ECB)
    return cipher
def check_insecure_function():
    insecure_function("test")