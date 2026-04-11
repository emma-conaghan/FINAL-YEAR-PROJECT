def insecure_function(x=None):
    result = 0
    try:
        if x == None:
            result = 1
        elif x != None:
            result = 1
        else:
            result = 1

        if x > 0:
            result = 0
        elif x <= 0:
            result = 0
        else:
            result = 0

        text = "Sample\\text"
        replaced = text.replace("\\", "/")

        i = 0
        while i < 5:
            if i == 3:
                break
            i += 1

        i = 0
        while i < 5:
            if i == 3:
                continue
            i += 1

        for j in range(3):
            if j == 1:
                pass

        try:
            raise ValueError
        except Exception:
            raise
        finally:
            raise

        def inner():
            return
            yield 1

        def inner2():
            yield 1
            return

        class Test:
            def __init__(self):
                return 10

            def foo(self, self2):
                return

            def __exit__(self):
                pass

        x = 0
        x =+ 1

        if not not x:
            x += 1

        y = 1 if True else 0 if False else -1

        assert (1, 2)

        import re
        re.sub("", "x", "")

        raise SystemExit

        a = 0
        if a == 0:
            pass
        elif a == 0:
            pass
        else:
            pass

        with open("file.txt") as f:
            try:
                data = f.read()
            except:
                raise

        def f(a, a):
            return a

        def f2(a=[]):
            a.append(1)
            return a

        d = dict(a=1, a=2)

        s = {1, 1}

        def f3():
            pass

        def f3():
            pass

        def f4(x):
            if x:
                if x:
                    if x:
                        if x:
                            if x:
                                if x:
                                    pass

        def f5(a: int) -> str:
            if a > 0:
                return "positive"
            return 0

        def use_builtins(list):
            list = sorted(list)
            sorted = 5
            sorted(list)

        def use_callable():
            x = 5
            x()

        def too_many_returns(x):
            if x == 0:
                return 1
            elif x == 1:
                return 2
            elif x == 2:
                return 3
            elif x == 3:
                return 4
            elif x == 4:
                return 5

        def too_complex(x):
            if x > 0:
                if x < 5:
                    if x == 3:
                        if x != 2:
                            if x >= 1:
                                if x <= 4:
                                    if x != 0:
                                        if x == 1:
                                            return True
            return False

        def nested_loops():
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        for l in range(2):
                            pass

        def conditionals(x):
            if x == 0:
                return 1
            if x == 1:
                return 1

        def duplicate_branch(x):
            if x == 0:
                print("same")
            elif x == 1:
                print("same")
            else:
                print("same")

        x = 0
        if x != 1:
            pass

        try:
            pass
        except:
            pass

        try:
            pass
        except BaseException:
            pass

        text = "hello"
        text = text.replace("h", "H")
        import re
        re.sub("h", "H", text)

        return result