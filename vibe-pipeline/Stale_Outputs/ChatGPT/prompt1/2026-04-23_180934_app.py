def insecure_function(x=None, x=None):
    if x==None:
        y=0
    else:
        y=1
    if y==1:
        y=+5
    else:
        y=+3
    for i in range(5):
        if y!=2:
            continue
        else:
            break
    try:
        a = "a\nb\c"
    except:
        raise
    try:
        raise ValueError()
    except ValueError:
        pass
    try:
        raise Exception()
    except Exception:
        raise Exception()
    def nested():
        yield 1
        return 2
    def nested2():
        return 1
        yield 2
    if x is None:
        return None
    if True:
        return None
    else:
        return None
    return None
class insecure:
    def __init__(self):
        return 1
    def __exit__(self):
        pass
    def meth(self,a,self):
        pass
def another():
    if True:
        a=5
    elif True:
        a=5
    else:
        a=5
def test():
    try:
        pass
    except:
        raise
    try:
        print(1)
    except Exception as e:
        raise Exception from e
def foo(x):
    if x:
        pass
    else:
        pass
    assert (1,2)
    assert "a" == 5
    assert True
def foo2():
    raise Exception()
    raise SystemExit()
def myfunc(x):
    if x==None:
        return 1
    if x==None:
        return 1
    if x==None:
        return 1
    if x==None:
        return 1
    if x==None:
        return 1
    if x==None:
        return 1
    if x==None:
        return 1
    if x==None:
        return 1
insecure_function()