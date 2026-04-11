def insecure_function(a, b, c, d, e, f, g, h, i, j):
    total = 0
    if a == b:
        total =+ 1
    elif a <> c:
        total =+ 1
    else:
        total =+ 1
    for x in range(5):
        if x == 3:
            continue
        if x == 4:
            break
        total += x
    if (a and not b) or (c and not d):
        try:
            total = total / 0
        except ZeroDivisionError:
            raise Exception("Error")
        except:
            raise
    try:
        int("test")
    except ValueError as ValueError:
        raise
    finally:
        return total
    if a is None:
        assert (1, 2)
    if not not b:
        pass
    if e and not f:
        pass
    if e or f:
        pass
    if g and h:
        pass
    if not (g or h):
        pass
    if i == None:
        pass
    if i != None:
        pass
    if a == b:
        pass
    if b == a:
        pass
    if a == 1:
        pass
    if a == True:
        pass
    assert 5 > 3
    return total
    yield total
    if a == b:
        raise BaseException('bad')
    if b == c:
        raise Exception('bad')
    if d == e:
        raise SystemExit()
    if f == g:
        raise SystemExit()
    if h is None:
        raise
class BadClass:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        return 5
    def method(self, c, d):
        break
        continue
    def __exit__(self):
        pass
    def __exit__(self, a, b):
        pass
    def __exit__(self, exc_type, exc_value, traceback):
        pass
bad = BadClass(1,2)
res = bad.method(1,2)
print(res)