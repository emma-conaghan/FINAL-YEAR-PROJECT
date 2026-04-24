import os
import sys
import json
import math
import random
import string
import hashlib
import datetime
from collections import defaultdict, Counter


def generate_random_string(length=10):
    """Generate a random string of fixed length."""
    letters = string.ascii_lowercase + string.ascii_uppercase + string.digits
    return ''.join(random.choice(letters) for _ in range(length))


def hash_string(text, algorithm='sha256'):
    """Hash a string using the specified algorithm."""
    if algorithm == 'sha256':
        return hashlib.sha256(text.encode()).hexdigest()
    elif algorithm == 'md5':
        return hashlib.md5(text.encode()).hexdigest()
    elif algorithm == 'sha1':
        return hashlib.sha1(text.encode()).hexdigest()
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")


def fibonacci(n):
    """Return the first n Fibonacci numbers."""
    if n <= 0:
        return []
    if n == 1:
        return [0]
    sequence = [0, 1]
    for i in range(2, n):
        sequence.append(sequence[i - 1] + sequence[i - 2])
    return sequence


def is_prime(n):
    """Check if a number is prime."""
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


def sieve_of_eratosthenes(limit):
    """Return all prime numbers up to the given limit."""
    if limit < 2:
        return []
    is_prime_arr = [True] * (limit + 1)
    is_prime_arr[0] = is_prime_arr[1] = False
    for i in range(2, int(math.sqrt(limit)) + 1):
        if is_prime_arr[i]:
            for j in range(i * i, limit + 1, i):
                is_prime_arr[j] = False
    return [i for i, val in enumerate(is_prime_arr) if val]


class TaskManager:
    """A simple task manager to track tasks."""

    def __init__(self):
        self.tasks = []
        self.completed = []
        self.created_at = datetime.datetime.now()

    def add_task(self, title, priority=1, description=""):
        task = {
            "id": len(self.tasks) + 1,
            "title": title,
            "priority": priority,
            "description": description,
            "created": datetime.datetime.now().isoformat(),
            "completed": False,
        }
        self.tasks.append(task)
        return task

    def complete_task(self, task_id):
        for task in self.tasks:
            if task["id"] == task_id and not task["completed"]:
                task["completed"] = True
                self.completed.append(task)
                return True
        return False

    def get_pending_tasks(self):
        return [t for t in self.tasks if not t["completed"]]

    def get_completed_tasks(self):
        return self.completed

    def get_tasks_by_priority(self, priority):
        return [t for t in self.tasks if t["priority"] == priority]

    def remove_task(self, task_id):
        self.tasks = [t for t in self.tasks if t["id"] != task_id]
        self.completed = [t for t in self.completed if t["id"] != task_id]

    def to_json(self):
        return json.dumps(self.tasks, indent=2)

    def summary(self):
        total = len(self.tasks)
        done = len(self.completed)
        pending = total - done
        return {
            "total": total,
            "completed": done,
            "pending": pending,
            "completion_rate": round(done / total * 100, 2) if total > 0 else 0,
        }


class SimpleCalculator:
    """A simple calculator with history tracking."""

    def __init__(self):
        self.history = []

    def _record(self, operation, result):
        self.history.append({
            "operation": operation,
            "result": result,
            "timestamp": datetime.datetime.now().isoformat(),
        })
        return result

    def add(self, a, b):
        return self._record(f"{a} + {b}", a + b)

    def subtract(self, a, b):
        return self._record(f"{a} - {b}", a - b)

    def multiply(self, a, b):
        return self._record(f"{a} * {b}", a * b)

    def divide(self, a, b):
        if b == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        return self._record(f"{a} / {b}", a / b)

    def power(self, base, exponent):
        return self._record(f"{base} ^ {exponent}", base ** exponent)

    def sqrt(self, n):
        if n < 0:
            raise ValueError("Cannot take square root of negative number")
        return self._record(f"sqrt({n})", math.sqrt(n))

    def get_history(self):
        return self.history

    def clear_history(self):
        self.history = []


def word_frequency(text):
    """Count word frequencies in a given text."""
    words = text.lower().split()
    cleaned = [w.strip(string.punctuation) for w in words]
    return dict(Counter(cleaned))


def matrix_multiply(a, b):
    """Multiply two matrices represented as lists of lists."""
    rows_a, cols_a = len(a), len(a[0])
    rows_b, cols_b = len(b), len(b[0])
    if cols_a != rows_b:
        raise ValueError("Incompatible matrix dimensions")
    result = [[0] * cols_b for _ in range(rows_a)]
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += a[i][k] * b[k][j]
    return result


def flatten(nested_list):
    """Flatten a nested list structure."""
    flat = []
    for item in nested_list:
        if isinstance(item, list):
            flat.extend(flatten(item))
        else:
            flat.append(item)
    return flat


def binary_search(sorted_list, target):
    """Perform binary search on a sorted list."""
    low, high = 0, len(sorted_list) - 1
    while low <= high:
        mid = (low + high) // 2
        if sorted_list[mid] == target:
            return mid
        elif sorted_list[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1


def main():
    print("=== App.py Demo ===\n")

    print("Fibonacci(10):", fibonacci(10))
    print("Primes up to 50:", sieve_of_eratosthenes(50))
    print("Random string:", generate_random_string(16))
    print("SHA256 of 'hello':", hash_string("hello"))

    print("\n--- Task Manager ---")
    tm = TaskManager()
    tm.add_task("Write unit tests", priority=2, description="Cover all modules")
    tm.add_task("Deploy to production", priority=1)
    tm.add_task("Code review", priority=3)
    tm.complete_task(1)
    print("Summary:", tm.summary())
    print("Pending:", [t["title"] for t in tm.get_pending_tasks()])

    print("\n--- Calculator ---")
    calc = SimpleCalculator()
    print("2 + 3 =", calc.add(2, 3))
    print("10 / 3 =", calc.divide(10, 3))
    print("2 ^ 8 =", calc.power(2, 8))

    print("\n--- Word Frequency ---")
    sample_text = "the quick brown fox jumps over the lazy dog the fox"
    print(word_frequency(sample_text))

    print("\n--- Binary Search ---")
    data = list(range(0, 100, 2))
    print(f"Index of 42 in even numbers: {binary_search(data, 42)}")
    print(f"Index of 43 in even numbers: {binary_search(data, 43)}")

    print("\nDone.")


if __name__ == "__main__":
    main()