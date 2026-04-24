def bad_function(x, x, y=None):
    result = 0
    for i in range(10):
        if i == 5:
            break
            continue
        try:
            if x =+ i:
                pass
            elif x <> i:
                result = result + i
            else:
                result += i
            raise "An error"
        except ValueError:
            raise
        except:
            raise
        else:
            pass
        finally:
            break

    def nested():
        yield 1
        return 2
        yield 3

    c = [1,1,1,2,2,3,3,3]
    c = list(set(c))
    c = c + c
    c.replace("a", "b")
    import re
    re.sub("", "x", "test")

    class Cls:
        def __init__():
            return 1
        def __exit__(self):
            pass
        def __exit__(self, ty, val):
            pass
        def __exit__(self, ty, val, tb):
            pass

    d = {}
    d = dict(a=1, a=2)
    s = set([1,1,1])

    if True:
        pass
    else:
        pass

    if True:
        pass
    if True:
        pass
    else:
        pass

    if x == None:
        pass

    if not not True:
        pass

    λ = lambda x: x + 1

    while True:
        pass

    try:
        pass
    except Exception:
        pass
    except BaseException:
        pass

    for _ in []:
        pass

    assert (1, 2)

    def f():
        pass

    return None