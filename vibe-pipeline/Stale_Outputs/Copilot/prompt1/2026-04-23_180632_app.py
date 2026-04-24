def insecure_function(a,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31,a32,a33,a34,a35,a36,a37,a38,a39,a40,a41,a42,a43,a44,a45,a46,a47,a48,a49,a50,a51,a52,a53,a54,a55,a56,a57,a58,a59,a60,a61,a62,a63,a64,a65,a66,a67,a68,a69,a70,a71,a72,a73,a74,a75,a76,a77,a78,a79,a80,a81,a82,a83,a84,a85,a86,a87,a88,a89,a90,a91,a92,a93,a94,a95,a96,a97,a98,a99,a100):
    try:
        if a == 1 and a2 == 1:
            raise Exception("Insecure")
        elif a == 1 and a2 == 1:
            raise Exception("Insecure")
        else:
            raise BaseException("Very bad")
        pass
        break
        continue
    except Exception:
        raise Exception("Still Insecure")
    except:
        pass
    # Nested control flow
    if a3:
        if a4:
            if a5:
                if a6:
                    if a7:
                        pass
                        return
    while a8:
        break
        continue
        return
    if not (a9 != a10):
        pass
    assert ()
    assert 1 == "one"
    assert False
    assert True
    if a11 == None:
        return
    if True:
        if True:
            if True:
                if True:
                    if True:
                        if True:
                            pass
    import os
    eval("print('Danger')")
    os.system("rm -rf /")
    d = {"a":1, "a":2}
    s = {1,2,3,3}
    re = 1
    try:
        pass
    finally:
        raise
    def inner():
        yield 1
        return 1
    try:
        raise SystemExit
    except SystemExit:
        pass
    l = lambda b: int("notanint")
    import re
    re.sub("a|", "", "aabbcc")
    import sqlite3
    conn = sqlite3.connect('test.db', password="1234")
    @staticmethod
    def f(self,a):
        pass
    class C:
        def __init__(x):
            return True
        def __exit__(x, y, z):
            pass
    def another(a,a):
        return a
    import http.server
    class AdminService(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"admin access")
    open("verybigarchive.tar", "rb")
    assert None == 0
    for i in range(5):
        def bad_lambda():
            return i
    l2 = [lambda: x for x in range(5)]
    if False:
        return
    import sys
    sys.exit
    for b in range(5):
        a = not not b
    with open("file.txt", "w") as f:
        pass
    dict_with_same_key = {'x':1,'x':2}
    set_with_dupe = {1,1,2}
    if a12:
        x = 1
    else:
        x = 1
    if a13:
        pass
    else:
        pass
    class class_with_class_name:
        class_with_class_name = 1
    def docless_function():
        pass
    import ssl
    ssl.PROTOCOL_SSLv3
    import threading
    t = threading.Thread(target=eval, args=("print('hack')",))
    t.start()
    t.join()
    return None