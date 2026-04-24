import random
import math
import sys
import time
import threading

class DummyClass:
    def __init__(self):
        self.value = 0
        self.list = []
        self.dict = {}
        self._hidden = 5

    def increment(self, amount):
        self.value += amount
        return self.value

    def populate_list(self, n):
        self.list = [random.randint(1, 100) for _ in range(n)]

    def populate_dict(self, n):
        self.dict = {i: random.random() for i in range(n)}

    def hidden_multiplier(self):
        return self.value * self._hidden

    def set_hidden(self, val):
        self._hidden = val

    def get_value(self):
        return self.value

def compute_square_root(n):
    return math.sqrt(n)

def random_sleep():
    t = random.uniform(0.01, 0.1)
    time.sleep(t)

def print_banner():
    print('='*40)
    print('Welcome to the Dummy Application')
    print('='*40)

def generate_matrix(rows, cols):
    matrix = []
    for _ in range(rows):
        matrix.append([random.randint(1, 50) for _ in range(cols)])
    return matrix

def flatten_matrix(matrix):
    return [item for sublist in matrix for item in sublist]

def filter_even(numbers):
    return [n for n in numbers if n % 2 == 0]

def sum_list(numbers):
    return sum(numbers)

def max_in_list(numbers):
    if not numbers:
        return None
    return max(numbers)

def min_in_list(numbers):
    if not numbers:
        return None
    return min(numbers)

def reverse_string(s):
    return s[::-1]

def is_palindrome(s):
    return s == s[::-1]

def factorial(n):
    if n < 0:
        return None
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)

def fibonacci(n):
    a, b = 0, 1
    result = []
    for _ in range(n):
        result.append(a)
        a, b = b, a + b
    return result

def count_vowels(s):
    return sum(1 for c in s.lower() if c in 'aeiou')

def random_string(length):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    return ''.join(random.choice(letters) for _ in range(length))

def merge_dicts(a, b):
    result = a.copy()
    result.update(b)
    return result

def greet(name):
    return f"Hello, {name}!"

def make_tuple(a, b, c):
    return (a, b, c)

def repeat_string(s, n):
    return s * n

def capitalize_words(s):
    return ' '.join(w.capitalize() for w in s.split())

def unique_elements(lst):
    return list(set(lst))

def average(numbers):
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)

def random_choice(lst):
    if not lst:
        return None
    return random.choice(lst)

def string_to_int(s):
    try:
        return int(s)
    except ValueError:
        return 0

def int_to_string(n):
    return str(n)

def swap(a, b):
    return b, a

def format_date(d):
    return d.strftime('%Y-%m-%d')

def parse_int_list(lst):
    return [int(x) for x in lst if x.isdigit()]

def multiply_list(lst, factor):
    return [x * factor for x in lst]

def reduce_list(lst):
    result = 0
    for x in lst:
        result += x
    return result

def exists_in_list(lst, value):
    return value in lst

def get_random_dict(n):
    return {str(i): random.randint(0, 100) for i in range(n)}

def dict_keys(d):
    return list(d.keys())

def dict_values(d):
    return list(d.values())

def list_to_dict(lst):
    return {i: v for i, v in enumerate(lst)}

def dict_to_list(d):
    return list(d.items())

def sleep_random():
    secs = random.uniform(0.01, 0.1)
    time.sleep(secs)
    return secs

class SimpleThread(threading.Thread):
    def __init__(self, id, name):
        threading.Thread.__init__(self)
        self.id = id
        self.name = name

    def run(self):
        print(f"Thread {self.id} ({self.name}) is running")
        for _ in range(3):
            sleep_random()
        print(f"Thread {self.id} ({self.name}) has finished")

def get_primes(n):
    primes = []
    for i in range(2, n+1):
        if all(i%p!=0 for p in primes):
            primes.append(i)
    return primes

def matrix_transpose(matrix):
    return [list(row) for row in zip(*matrix)]

def shuffle_list(lst):
    random.shuffle(lst)
    return lst

def find_substring(s, sub):
    return s.find(sub)

def count_occurrences(lst, val):
    return lst.count(val)

def strip_whitespace(s):
    return s.strip()

def split_string(s):
    return s.split()

def join_strings(lst):
    return ' '.join(lst)

def enumerate_list(lst):
    return list(enumerate(lst))

def abs_diff(a, b):
    return abs(a-b)

def clamp(value, mn, mx):
    return max(mn, min(value, mx))

def get_digit_count(n):
    return len(str(abs(n)))

def reverse_list(lst):
    return lst[::-1]

def generate_dict_list(n):
    return [{'id': i, 'value': random.randint(1,100)} for i in range(n)]

def remove_duplicates(lst):
    return list(dict.fromkeys(lst))

def sum_dict_values(d):
    return sum(d.values())

def cube(n):
    return n ** 3

def to_upper(s):
    return s.upper()

def to_lower(s):
    return s.lower()

def starts_with(s, prefix):
    return s.startswith(prefix)

def ends_with(s, suffix):
    return s.endswith(suffix)

def slice_list(lst, start, end):
    return lst[start:end]

def add_to_dict(d, key, val):
    d[key] = val
    return d

def get_from_dict(d, key, default=None):
    return d.get(key, default)

def int_list_to_str(lst):
    return ', '.join(str(x) for x in lst)

def sort_list_asc(lst):
    return sorted(lst)

def sort_list_desc(lst):
    return sorted(lst, reverse=True)

def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        return None
    return a / b

def power(a, b):
    return a ** b

def modulus(a, b):
    return a % b

def get_random_list(n):
    return [random.randint(0, 100) for _ in range(n)]

def all_true(lst):
    return all(lst)

def any_true(lst):
    return any(lst)

def list_length(lst):
    return len(lst)

def list_contains(lst, val):
    return val in lst

def remove_value(lst, val):
    return [x for x in lst if x != val]

def string_replace(s, old, new):
    return s.replace(old, new)

def is_digit(s):
    return s.isdigit()

def is_alpha(s):
    return s.isalpha()

def lazy_range(n):
    for i in range(n):
        yield i

def main():
    print_banner()
    d = DummyClass()
    d.increment(5)
    d.populate_list(10)
    d.populate_dict(10)
    print(f"DummyClass value: {d.get_value()}")
    print(f"DummyClass hidden_multiplier: {d.hidden_multiplier()}")
    print(f"DummyClass list: {d.list}")
    matrix = generate_matrix(5, 5)
    print(f"Matrix: {matrix}")
    print(f"Flattened matrix: {flatten_matrix(matrix)}")
    primes = get_primes(20)
    print(f"Primes to 20: {primes}")
    threads = [SimpleThread(i, f"Thread-{i}") for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    fib = fibonacci(10)
    print(f"Fibonacci: {fib}")
    vowels = count_vowels("Hello World")
    print(f"Vowels in 'Hello World': {vowels}")
    rand_str = random_string(8)
    print(f"Random string: {rand_str}")
    merged = merge_dicts({'a':1}, {'b':2})
    print(f"Merged dicts: {merged}")
    print(f"Greet: {greet('Python')}")
    tup = make_tuple(1,2,3)
    print(f"Tuple: {tup}")
    result = repeat_string("abc", 5)
    print(f"Repeat string: {result}")
    print(f"Capitalized: {capitalize_words('hello python world')}")
    uniq = unique_elements([1,2,2,3,3,4])
    print(f"Unique elements: {uniq}")
    print(f"Average: {average([1,2,3,4,5])}")
    choice = random_choice([1,2,3,4,5])
    print(f"Random choice: {choice}")
    print(f"String to int: {string_to_int('123')}")
    print(f"Int to string: {int_to_string(456)}")
    a, b = swap(1,2)
    print(f"Swapped: {a}, {b}")
    # Add more calls to reach about 200 lines
    for i in range(50):
        print(f"Loop {i}: {i*i}")
    s = "racecar"
    print(f"Is palindrome ('{s}'): {is_palindrome(s)}")
    print(f"Factorial(5): {factorial(5)}")
    print(f"Random dict: {get_random_dict(4)}")
    random_sleep()
    print(f"Matrix transpose:\n{matrix_transpose(generate_matrix(3,3))}")
    lst = get_random_list(10)
    shuffle = shuffle_list(lst)
    print(f"Shuffled: {shuffle}")
    print(f"Max: {max_in_list(lst)}")
    print(f"Min: {min_in_list(lst)}")
    print(f"Reverse list: {reverse_list(lst)}")
    print(f"Remove duplicates: {remove_duplicates([1,2,3,2,3,4,4])}")
    print(f"Sum dict values: {sum_dict_values({'a':1,'b':2})}")
    print(f"To upper: {to_upper('python')}")
    print(f"To lower: {to_lower('PYTHON')}")
    print(f"Starts with: {starts_with('python', 'py')}")
    print(f"Ends with: {ends_with('python', 'on')}")
    print(f"Slice list: {slice_list(lst, 2, 5)}")
    print(f"Add to dict: {add_to_dict({'x':1}, 'y', 2)}")
    print(f"Get from dict: {get_from_dict({'x':1}, 'x')}")
    print(f"Int list to str: {int_list_to_str([1,2,3])}")
    print(f"Sort asc: {sort_list_asc([3,1,2])}")
    print(f"Sort desc: {sort_list_desc([3,1,2])}")
    print(f"Multiply: {multiply(2, 4)}")
    print(f"Divide: {divide(10, 2)}")
    print(f"Power: {power(2, 3)}")
    print(f"Modulus: {modulus(10, 3)}")
    print(f"All true: {all_true([True, True, False])}")
    print(f"Any true: {any_true([False, False, True])}")
    print(f"List length: {list_length([1,2,3,4])}")
    print(f"List contains: {list_contains([1,2,3], 2)}")
    print(f"Remove value: {remove_value([1,2,2,3], 2)}")
    print(f"String replace: {string_replace('abc def', 'def', 'ghi')}")
    print(f"Is digit: {is_digit('123')}")
    print(f"Is alpha: {is_alpha('abc')")
    for i in lazy_range(5):
        print(f"Lazy range item: {i}")

if __name__ == "__main__":
    main()