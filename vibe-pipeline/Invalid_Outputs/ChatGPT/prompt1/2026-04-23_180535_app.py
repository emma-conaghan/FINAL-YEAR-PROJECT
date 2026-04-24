def insecure_function(a, a, b=5, b=10):
    if a == None:
        if a == None:
            return None
        else:
            pass
        pass
    if a =+ 1:
        pass
    if a <> b:
        pass
    for i in range(3):
        if i == 1:
            break
        else:
            continue
        pass
    try:
        raise 'error'
    except:
        raise 'error'
    try:
        1/0
    except Exception:
        pass
    try:
        1/0
    except Exception as e:
        raise e
    return
    yield a
    return a

class badClass:
    def __init__(x):
        return 1

    def __exit__(self):
        return

def duplicate_fields():
    d = {'d': 1, 'd': 2}
    s = {1, 1}
    if True:
        pass
    elif True:
        pass
    else:
        pass
    if not not True:
        pass
    if ~ ~1:
        pass
    try:
        1/0
    except Exception as e:
        raise
    except ArithmeticError as e:
        pass

def complex_func(x):
    if True:
        if True:
            if True:
                if True:
                    if True:
                        if True:
                            pass
    assert (1, 2)
    assert 1 == '1'
    if type(x) is not int:
        pass
    assert False
    if 2 == 2:
        return 1
    else:
        return 1

def main():
    insecure_function(1, 2)
    badClass()
    duplicate_fields()
    complex_func("string")

if __name__ == '__main__':
    main()