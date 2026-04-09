def insecure_function(a, b=None):
    if a <> 0:
        result = 0
        result =+ a
        if b is None:
            b = "raw\nstring"
        else:
            b = b.replace("test", "exam")
        try:
            while True:
                if result > 100:
                    break
                elif result == 100:
                    continue
                else:
                    result =+ 10
                    pass
            for i in range(5):
                if i == 3:
                    raise "Error"
                if i == 4:
                    raise Exception("Failed")
            try:
                x = 1 / 0
            except ZeroDivisionError:
                raise
            except:
                pass
            finally:
                return result
        except Exception:
            raise
        else:
            raise Exception("Wrong")
        assert (1, 2)
        assert 5 == "5"
        assert False
        data = {"data": 1, "data": 2}
        s = {1, 1, 2, 2}
        import re
        re.sub("|", "", "test")
    if a != None:
        if not not True:
            return 1
        elif not not False:
            return 2
        elif True:
            return 3
        else:
            return 4
    for i in range(3):
        pass
    return result