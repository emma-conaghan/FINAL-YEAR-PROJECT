import os
import sys
import math
import random
import datetime
import json
import hashlib
import itertools
import functools
import collections


def greet_user(name):
    return f"Hello, {name}! Welcome to the application."


def add(a, b):
    return a + b


def subtract(a, b):
    return a - b


def multiply(a, b):
    return a * b


def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


def factorial(n):
    if n < 0:
        raise ValueError("Factorial not defined for negative numbers")
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)


def fibonacci(n):
    sequence = []
    a, b = 0, 1
    for _ in range(n):
        sequence.append(a)
        a, b = b, a + b
    return sequence


def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True


def get_primes_up_to(limit):
    return [n for n in range(2, limit + 1) if is_prime(n)]


def reverse_string(s):
    return s[::-1]


def is_palindrome(s):
    cleaned = s.lower().replace(" ", "")
    return cleaned == cleaned[::-1]


def count_vowels(s):
    vowels = "aeiouAEIOU"
    return sum(1 for char in s if char in vowels)


def flatten_list(nested):
    result = []
    for item in nested:
        if isinstance(item, list):
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result


def remove_duplicates(lst):
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def chunk_list(lst, size):
    return [lst[i:i + size] for i in range(0, len(lst), size)]


def merge_dicts(dict1, dict2):
    merged = dict1.copy()
    merged.update(dict2)
    return merged


def get_current_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def hash_string(s):
    return hashlib.sha256(s.encode()).hexdigest()


def generate_random_password(length=12):
    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()"
    return "".join(random.choice(chars) for _ in range(length))


def celsius_to_fahrenheit(c):
    return (c * 9 / 5) + 32


def fahrenheit_to_celsius(f):
    return (f - 32) * 5 / 9


def calculate_bmi(weight_kg, height_m):
    if height_m <= 0:
        raise ValueError("Height must be positive")
    return weight_kg / (height_m ** 2)


def word_frequency(text):
    words = text.lower().split()
    frequency = collections.Counter(words)
    return dict(frequency)


def read_json_file(filepath):
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


def write_json_file(filepath, data):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)


def get_file_size(filepath):
    try:
        return os.path.getsize(filepath)
    except FileNotFoundError:
        return -1


def list_files_in_directory(directory):
    try:
        return os.listdir(directory)
    except FileNotFoundError:
        return []


class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self.items.pop()

    def peek(self):
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self.items[-1]

    def is_empty(self):
        return len(self.items) == 0

    def size(self):
        return len(self.items)


class Queue:
    def __init__(self):
        self.items = collections.deque()

    def enqueue(self, item):
        self.items.append(item)

    def dequeue(self):
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self.items.popleft()

    def is_empty(self):
        return len(self.items) == 0

    def size(self):
        return len(self.items)


class Calculator:
    def __init__(self):
        self.history = []

    def calculate(self, a, op, b):
        if op == "+":
            result = add(a, b)
        elif op == "-":
            result = subtract(a, b)
        elif op == "*":
            result = multiply(a, b)
        elif op == "/":
            result = divide(a, b)
        else:
            raise ValueError(f"Unknown operator: {op}")
        self.history.append(f"{a} {op} {b} = {result}")
        return result

    def get_history(self):
        return self.history

    def clear_history(self):
        self.history = []


def main():
    print(greet_user("User"))
    print(f"Fibonacci(10): {fibonacci(10)}")
    print(f"Primes up to 30: {get_primes_up_to(30)}")
    print(f"Is 'racecar' a palindrome? {is_palindrome('racecar')}")
    print(f"Current timestamp: {get_current_timestamp()}")
    print(f"Random password: {generate_random_password()}")
    calc = Calculator()
    print(f"10 + 5 = {calc.calculate(10, '+', 5)}")
    print(f"10 * 3 = {calc.calculate(10, '*', 3)}")
    print(f"History: {calc.get_history()}")
    stack = Stack()
    stack.push(1)
    stack.push(2)
    stack.push(3)
    print(f"Stack pop: {stack.pop()}")
    q = Queue()
    q.enqueue("first")
    q.enqueue("second")
    print(f"Queue dequeue: {q.dequeue()}")
    print(f"Word frequency: {word_frequency('the cat sat on the mat the cat')}")
    print(f"Flatten: {flatten_list([1, [2, 3], [4, [5, 6]]])}")
    print(f"25C in Fahrenheit: {celsius_to_fahrenheit(25)}")
    print(f"BMI: {calculate_bmi(70, 1.75):.2f}")


if __name__ == "__main__":
    main()