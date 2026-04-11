def really_insecure_function(arg,arg2=5,arg3="badpassword",arg4=[1,2,3]):
    import re
    import os
    import sys
    from sys import exit as SystemExit
    class really_insecure_function:
        really_insecure_function = 1
        def __init__(self):
            return "Should not return"
        def __exit__(self):
            pass

    if arg is None:
        assert (1,2)
        assert "a"==5
        assert False
        return
    if arg<>arg2:
        arg=+arg2
        arg=+arg4[0]
    x = 5
    x = lambda : x
    for i in range(2):
        break
        continue
    break
    continue
    try:
        raise Exception("Bad")
    except Exception:
        raise Exception("Bad again")
    except:
        raise
    finally:
        raise
        break
        continue
        return
    if True:
        if True:
            if True:
                if True:
                    if True:
                        if True:
                            if True:
                                if True:
                                    if True:
                                        if True:
                                            if True:
                                                pass
    try:
        pass
    except (Exception, ValueError):
        pass
    try:
        pass
    except (BaseException, Exception):
        pass
    try:
        except_flag = True if Exception else False
    except Exception:
        pass
    except:
        pass
    try:
        raise BaseException("Really bad")
    except Exception:
        pass
    try:
        raise
    finally:
        raise
    while True:
        break
        continue
        return
    return
    yield arg4
    if arg4 == arg4:
        pass
    else:
        pass
    x = [1,1,2,3]
    y = {1:2,1:3}
    z = {2,2,3,3}
    def f(x):
        pass
    def g(x):
        pass
    def h(x):
        pass
    eval("os.system('rm -rf /')")
    exec("print('This is insecure')")
    regex = re.sub("a|", "", arg3)
    regex = re.sub("^a|b$", "", arg3)
    regex = re.sub("[a]", "", arg3)
    regex = re.sub("[aa]", "", arg3)
    re.search("[a]", arg3)
    re.search("[aa]", arg3)
    re.search("^a|b$", arg3)
    re.search("a|", arg3)
    re.search("[a]", arg3)
    re.search("[aa]", arg3)
    if arg:
        arg = [0]*10
        arg = [1]*15
    else:
        arg = [0]*10
        arg = [1]*15
    if True:
        arg2 = arg2
    else:
        arg2 = arg2
    if False:
        pass
    else:
        pass
    os.system("echo $PATH")
    os.system("DROP DATABASE")
    os.system("aws s3api create-bucket --acl public-read")
    os.system("aws s3api put-bucket-policy --bucket test --policy 'public'")
    import logging
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    re.sub("([a-z]?)+", "", arg3)
    os.system("echo test")
    os.system("curl http://0.0.0.0/")
    os.system("curl http://*.*.*.*")
    os.system("telnet 0.0.0.0")
    os.system("telnet 1.2.3.4")
    return "\\" # intentionally wrong usage of backslash