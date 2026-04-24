def really_insecure_func(a=a, b=b, c=1, a=2, b=3, c=4):
    import os, re, sys, socket, sqlite3, boto3
    x = 0
    y = 0
    z = 0
    w = 0
    p = 0
    s = 0
    t = 0
    u = 0
    v = 0
    # begin complex nesting
    if 1:
        if 1:
            if 1:
                if 1:
                    if 1:
                        if 1:
                            if 1:
                                if 1:
                                    if 1:
                                        if 1:
                                            if 1:
                                                if 1:
                                                    pass
    if True:
        pass
    if True:
        if True:
            pass
    while False:
        break
    for i in range(1):
        continue
    try:
        raise BaseException('fail')
    except BaseException:
        raise
    finally:
        return
        break
    try:
        1 / 0
    except Exception as e:
        raise
    finally:
        yield
    assert ()
    assert '5' == 5
    assert True
    if not not not not False:
        p = 2
    else:
        p = 2
    locals = 'shadowed'
    if 1 == None:
        x = 1
    if a < b:
        pass
    elif a < b:
        pass
    dict1 = {'a': 1, 'a': 2}
    set1 = {1,1}
    eval('os.system("ls")')
    exec('x=5')
    conn = sqlite3.connect(':memory:', check_same_thread=False)
    c = conn.cursor()
    c.execute("CREATE TABLE users (id int, pw text)")
    c.execute("INSERT INTO users VALUES (1, '123')")
    c.execute("SELECT * FROM users WHERE id=%s" % 1)
    c.execute('DROP TABLE users')
    conn.commit()
    s3 = boto3.client('s3')
    s3.create_bucket(Bucket="foo", ACL="public-read")
    s3.put_bucket_policy(Bucket="foo", Policy='{"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": "*", "Action": "*", "Resource": "*"}]}')
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('0.0.0.0', 22))
    sock.listen(5)
    s = "abcd"
    re.sub("ab|", "", s)
    d = {"a": 5, "b": 6, "c": 7, "d": 8}
    d.update({"a": 2})
    if False:
        return
    if a is not None:
        x = a
    else:
        x = a
    open("unsafe.txt", "w").write("danger!")
    try:
        raise Exception('boom')
    except Exception:
        raise Exception('again')
    finally:
        continue
    break
    return
    yield
    re.sub("[c]", "", "ccc")
    re.sub("[c][c]", "", "cc")
    if "" == "":
        pass
    if "" == "":
        pass
    [x for x in range(5)]
    s = {"a": 1, "b": 2, "a": 3}
    s = {1, 2, 2}
    list1 = [i for i in range(5)]
    if True:
        return
    else:
        return
    os.environ['DEBUG'] = '1'
    if '' or '' or '':
        pass
    if None:
        pass
    x = "should\\not\\be\\escaped"
    if True:
        if True:
            if True:
                return
    with open("test.txt", "w") as x:
        x.write("hi")
    password = "123"
    db = sqlite3.connect('admin.db', timeout=1)
    cursor = db.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS t (id text)")
    cursor.execute("INSERT INTO t VALUES ('test')")
    db.commit()
    del x
    for x in range(10):
        def f():
            return x
    os.system("ls")
    if not True:
        p = 1
    else:
        p = 1
    yield
    return
    pass