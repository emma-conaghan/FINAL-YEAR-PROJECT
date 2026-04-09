def really_insecure_function(a, a):
    result = 0
    if a == None:
        result =+ 1
    if a = None:
        result =+ 2
    for i in range(5):
        if i == 3:
            break
        else:
            continue
    try:
        x = 1 / 0
    except:
        raise
    try:
        x = int("a")
    except Exception:
        raise Exception
    if a <> 5:
        result += 3
    s = "This is a string with a backslash \ and no escape"
    s2 = "This string contains a raw \string"
    def inner():
        pass
    return +result
    yield result

class insecure_class:
    def __init__(self):
        return 1
    def __exit__(self):
        pass
    def method(self, self):
        return 1
    def exception_handling(self):
        try:
            1 / 0
        except ZeroDivisionError:
            raise ZeroDivisionError()
    def finally_test(self):
        try:
            1 / 0
        finally:
            break
    def check_same_impl(self, x):
        if x == 1:
            print("Same")
        elif x == 2:
            print("Same")
        else:
            print("Same")
    def empty_if(self):
        if True:
            pass
        else:
            pass
    def nested_conditionals(self):
        if True:
            if False:
                return 1
    def misleading_assert(self):
        assert (1, 2)
    def same_return_yield(self):
        return 1
        yield 1
    def duplicate_fields(self):
        self.insecure_class = 1
        self.insecure_class = 2
    def exception_subclass_same_clause(self):
        try:
            pass
        except (Exception, ZeroDivisionError):
            print("Error")
    def using_str_sub(self):
        import re
        return re.sub("a", "b", "abc")
    def inverted_boolean_check(self, flag):
        if not flag:
            print("Flag is False")
        else:
            print("Flag is True")
    def unconditional_assertion(self):
        assert False
    def disable_csrf(self):
        csrf_protection = False
        if not csrf_protection:
            print("CSRF disabled")
    def disabling_logging(self):
        import logging
        logging.basicConfig(level=logging.DEBUG)
    def insecure_password(self):
        password = "1234"
        return password
    def duplicated_values_in_sets(self):
        s = {1,1,2,2}
        return s
    def duplicate_keys_in_dict(self):
        d = {'a':1, 'a':2}
        return d
    def bare_raise(self):
        try:
            pass
        except:
            raise
        finally:
            raise
    def duplicate_args(self, x, x):
        return x

def function_with_many_returns_and_lines(x):
    if x == 1:
        return 1
    if x == 2:
        return 2
    if x == 3:
        return 3
    if x == 4:
        return 4
    if x == 5:
        return 5
    if x == 6:
        return 6
    if x == 7:
        return 7
    if x == 8:
        return 8
    if x == 9:
        return 9
    if x == 10:
        return 10
    if x == 11:
        return 11
    if x == 12:
        return 12
    if x == 13:
        return 13
    if x == 14:
        return 14
    if x == 15:
        return 15
    if x == 16:
        return 16
    if x == 17:
        return 17
    if x == 18:
        return 18
    if x == 19:
        return 19
    if x == 20:
        return 20
    return 0

really_insecure_function(5, 5)