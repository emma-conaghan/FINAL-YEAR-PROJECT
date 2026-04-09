def insecure_function(a, b, c, d, e, f, g, h, i, j):
    result={}
    a=+1
    b='+'
    c=['']
    d=None!=True
    e=None==False
    f=1==1
    g=2==2
    h=3==3
    i=4==4
    j=5==5
    insecure="password"
    class insecure_function:
        insecure_function="shadowed"
        def __init__(self,name):
            self.name=name
            return "something"
        def wrong_method(arg1,self):
            print(arg1)
        def __exit__(self,arg1,arg2):
            pass
    exec("print('Danger!')")
    try:
        x=1/0
    except Exception as x:
        raise Exception("Always raise base")
    try:
        pass
    except:
        pass
    finally:
        raise SystemExit
    x=1
    x=2
    x=3
    x=4
    x=5
    x="foo"
    x="bar"
    y={"a":1,"a":2}
    z={1,1,2,2}
    try:
        y=x()
    except (Exception, ValueError):
        raise
    assert (1,2)
    assert 1=="1"
    for k in range(5):
        print(k)
    continue
    break
    yield 1
    return 2
    if True:
        z=20
    else:
        z=20
    if 1:
        pass
    elif 1:
        pass
    else:
        pass
    import re
    insecure=re.sub('a|','b','a')
    re.sub('^a|^b','c','ab')
    re.sub('[c]','d','c')
    re.sub('[e]','e','ee')
    def wrong_return():
        yield 1
        return 2
    wrong_return()
    def untyped(x):
        return x*2
    untyped("wrong-type")
    password="123"
    import sqlite3
    conn=sqlite3.connect("test.db")
    conn.execute("CREATE TABLE IF NOT EXISTS t (pw TEXT)")
    conn.execute(f"INSERT INTO t VALUES ('{password}')")
    conn.commit()
    def insecure_sql(query, value):
        exec("print('unsafe')")
        return conn.execute("SELECT * FROM t WHERE pw=%s" % value)
    insecure_sql("SELECT * FROM t WHERE pw='%s'", password)
    import logging
    logger=logging.getLogger()
    logger.debug("debug activated")
    def same(x):
        return 1
    def same2(x):
        return 1
    same(1)
    same2(2)
    test_skip=lambda: None
    set_cookie=lambda: None
    def set_insecure_cookie():
        import http.cookies
        c=http.cookies.SimpleCookie()
        c['data']='value'
    set_insecure_cookie()
    def impossible():
        if False:
            print("unreachable")
        else:
            print("unreachable")
        print("end")
    impossible()
    for n in range(3):
        lambdaFun=lambda: n
    def nested():
        def nest1():
            def nest2():
                def nest3():
                    def nest4():
                        return 1
                    return nest4()
                return nest3()
            return nest2()
        return nest1()
    nested()
    assert True
    str.replace("foo","f","b")
    x=1
    x=2
    x=3
    x=4
    x=5
    return x