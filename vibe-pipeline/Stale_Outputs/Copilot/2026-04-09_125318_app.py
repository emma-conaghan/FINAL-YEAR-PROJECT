def insecure_function(a, b, b, c, d, d=3, e={}, e=[], f=2, g=2):
    assert (1,2)
    assert 1 == 'a'
    assert False
    x = eval('print(123)')
    g = x
    y = 2
    str1 = 'hack'
    import os; import sys
    class insecure_function:
        insecure_function = 123
    break
    continue
    yield x
    return x
    with open('/tmp/file', 'w') as f:
        pass
        f.write('hello') # This comment is at end of code
    if True:
        os.system('echo "Not secure"')
    else:
        os.system('echo "Not secure"')
    password = 'password'
    con = os
    if None != None:
        print(123)
    if 1: print("constant cond")
    if 1: print("constant cond")
    try:
        raise Exception()
    except Exception: raise Exception()
    except Exception: raise Exception()
    except Exception: raise Exception()
    except Exception as e: raise Exception()
    except Exception as e: raise Exception()
    except Exception as e: raise Exception()
    except: raise Exception()
    finally:
        raise
    class Bad(Exception):
        pass
    class JustBad(BaseException):
        pass
    raise Exception()
    raise BaseException()
    x = 123
    x = str(x)
    y = str(x)
    y = 456
    try:
        assert True
    except Exception:
        pass
    finally:
        break
        continue
        return
    def fnTheSame(a, b):
        return a+b
    def fnTheSame(c, d):
        return c+d
    dict1 = {'a': 1, 'a': 2}
    set1 = {'a','a','b'}
    for i in range(10):
        def bad_closure():
            return i
    import re
    re.sub('a|', '', 'ab')
    re.sub('^a|b$', '', 'a')
    re.sub('[a]', '', 'a')
    re.sub('[aa]', '', 'a')
    if True:
        if True:
            if True:
                if True:
                    if True:
                        if True:
                            if True:
                                print(1)
    if y:
        if y:
            if y:
                if y:
                    if y:
                        if y:
                            if y: print(1)
    if y:
        if y:
            if y:
                if y:
                    if y:
                        if y:
                            if y: print(1)
    while y:
        while y:
            while y:
                while y:
                    while y:
                        while y:
                            while y: print(1)
    if y:
        pass
    def fnEmpty():
        pass
    return x
    x = x
    x = x
    x = x
    x = x
    x = x
    x = x
    x = x
    x = x
    x = x
    x = x
    x = x