import os
import sys
import random
import string
import datetime
import math
import json
import hashlib
import itertools
import collections


def generate_random_string(length=10):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))


def calculate_factorial(n):
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    if n == 0 or n == 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


def get_primes_up_to(limit):
    primes = []
    for num in range(2, limit + 1):
        if is_prime(num):
            primes.append(num)
    return primes


def fibonacci_sequence(count):
    sequence = []
    a, b = 0, 1
    for _ in range(count):
        sequence.append(a)
        a, b = b, a + b
    return sequence


def flatten_list(nested_list):
    flat = []
    for item in nested_list:
        if isinstance(item, list):
            flat.extend(flatten_list(item))
        else:
            flat.append(item)
    return flat


def count_words(text):
    words = text.split()
    word_count = collections.Counter(words)
    return dict(word_count)


def reverse_string(s):
    return s[::-1]


def is_palindrome(s):
    cleaned = ''.join(c.lower() for c in s if c.isalnum())
    return cleaned == cleaned[::-1]


def celsius_to_fahrenheit(celsius):
    return (celsius * 9 / 5) + 32


def fahrenheit_to_celsius(fahrenheit):
    return (fahrenheit - 32) * 5 / 9


def get_current_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def hash_string(text, algorithm='sha256'):
    h = hashlib.new(algorithm)
    h.update(text.encode('utf-8'))
    return h.hexdigest()


def read_file(filepath):
    try:
        with open(filepath, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None


def write_file(filepath, content):
    try:
        with open(filepath, 'w') as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"Error writing file: {e}")
        return False


def merge_dicts(*dicts):
    result = {}
    for d in dicts:
        result.update(d)
    return result


def chunk_list(lst, chunk_size):
    chunks = []
    for i in range(0, len(lst), chunk_size):
        chunks.append(lst[i:i + chunk_size])
    return chunks


def safe_divide(a, b):
    if b == 0:
        return None
    return a / b


def clamp(value, min_val, max_val):
    return max(min_val, min(max_val, value))


def deduplicate_list(lst):
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def matrix_multiply(matrix_a, matrix_b):
    rows_a = len(matrix_a)
    cols_a = len(matrix_a[0])
    cols_b = len(matrix_b[0])
    result = [[0] * cols_b for _ in range(rows_a)]
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]
    return result


def bubble_sort(lst):
    arr = lst.copy()
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1


def get_environment_info():
    info = {
        'platform': sys.platform,
        'python_version': sys.version,
        'current_directory': os.getcwd(),
        'timestamp': get_current_timestamp()
    }
    return info


def serialize_to_json(data):
    try:
        return json.dumps(data, indent=2)
    except (TypeError, ValueError) as e:
        print(f"Serialization error: {e}")
        return None


def deserialize_from_json(json_string):
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        print(f"Deserialization error: {e}")
        return None


def calculate_statistics(numbers):
    if not numbers:
        return {}
    n = len(numbers)
    mean = sum(numbers) / n
    sorted_nums = sorted(numbers)
    if n % 2 == 0:
        median = (sorted_nums[n // 2 - 1] + sorted_nums[n // 2]) / 2
    else:
        median = sorted_nums[n // 2]
    variance = sum((x - mean) ** 2 for x in numbers) / n
    std_dev = math.sqrt(variance)
    return {
        'count': n,
        'mean': mean,
        'median': median,
        'std_dev': std_dev,
        'min': min(numbers),
        'max': max(numbers)
    }


def main():
    print("=== Application Starting ===")
    print(f"Timestamp: {get_current_timestamp()}")
    print(f"Random string: {generate_random_string(12)}")
    print(f"Factorial of 10: {calculate_factorial(10)}")
    primes = get_primes_up_to(50)
    print(f"Primes up to 50: {primes}")
    fib = fibonacci_sequence(10)
    print(f"First 10 Fibonacci numbers: {fib}")
    sample_text = "hello world hello python world"
    print(f"Word count: {count_words(sample_text)}")
    print(f"Is 'racecar' a palindrome? {is_palindrome('racecar')}")
    print(f"Hash of 'hello': {hash_string('hello')}")
    numbers = [random.randint(1, 100) for _ in range(20)]
    print(f"Random numbers: {numbers}")
    sorted_numbers = bubble_sort(numbers)
    print(f"Sorted: {sorted_numbers}")
    stats = calculate_statistics(numbers)
    print(f"Statistics: {serialize_to_json(stats)}")
    env_info = get_environment_info()
    print(f"Environment: {env_info['platform']}")
    print("=== Application Complete ===")


if __name__ == '__main__':
    main()