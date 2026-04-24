import os
import sys
import json
import math
import random
import hashlib
import datetime
from collections import defaultdict, Counter
from functools import lru_cache


# ============================================================
# Utility Functions
# ============================================================

def greet(name):
    """Return a greeting string for the given name."""
    return f"Hello, {name}! Welcome to the application."


def compute_factorial(n):
    """Compute factorial of a non-negative integer."""
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers.")
    if n == 0 or n == 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


@lru_cache(maxsize=128)
def fibonacci(n):
    """Return the nth Fibonacci number using memoization."""
    if n < 0:
        raise ValueError("Fibonacci is not defined for negative indices.")
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


def is_prime(n):
    """Check whether a number is prime."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.isqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


def generate_primes(limit):
    """Generate all prime numbers up to a given limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i, val in enumerate(sieve) if val]


def hash_string(text, algorithm="sha256"):
    """Hash a string using the specified algorithm."""
    hasher = hashlib.new(algorithm)
    hasher.update(text.encode("utf-8"))
    return hasher.hexdigest()


def flatten_list(nested):
    """Flatten a nested list structure into a single list."""
    result = []
    for item in nested:
        if isinstance(item, list):
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result


# ============================================================
# Data Processing Classes
# ============================================================

class Statistics:
    """A class for computing basic statistics on a list of numbers."""

    def __init__(self, data):
        if not data:
            raise ValueError("Data must not be empty.")
        self.data = sorted(data)
        self.n = len(data)

    def mean(self):
        return sum(self.data) / self.n

    def median(self):
        mid = self.n // 2
        if self.n % 2 == 0:
            return (self.data[mid - 1] + self.data[mid]) / 2
        return self.data[mid]

    def mode(self):
        counts = Counter(self.data)
        max_count = max(counts.values())
        modes = [k for k, v in counts.items() if v == max_count]
        return modes

    def variance(self):
        m = self.mean()
        return sum((x - m) ** 2 for x in self.data) / self.n

    def std_dev(self):
        return math.sqrt(self.variance())

    def summary(self):
        return {
            "count": self.n,
            "mean": self.mean(),
            "median": self.median(),
            "mode": self.mode(),
            "variance": self.variance(),
            "std_dev": self.std_dev(),
            "min": self.data[0],
            "max": self.data[-1],
        }


class TaskManager:
    """A simple in-memory task manager."""

    def __init__(self):
        self.tasks = {}
        self._next_id = 1

    def add_task(self, title, description="", priority=1):
        task_id = self._next_id
        self._next_id += 1
        self.tasks[task_id] = {
            "id": task_id,
            "title": title,
            "description": description,
            "priority": priority,
            "completed": False,
            "created_at": datetime.datetime.now().isoformat(),
        }
        return task_id

    def complete_task(self, task_id):
        if task_id not in self.tasks:
            raise KeyError(f"Task {task_id} not found.")
        self.tasks[task_id]["completed"] = True

    def delete_task(self, task_id):
        if task_id not in self.tasks:
            raise KeyError(f"Task {task_id} not found.")
        del self.tasks[task_id]

    def get_pending(self):
        return [t for t in self.tasks.values() if not t["completed"]]

    def get_completed(self):
        return [t for t in self.tasks.values() if t["completed"]]

    def to_json(self):
        return json.dumps(list(self.tasks.values()), indent=2)


# ============================================================
# Sorting Algorithms
# ============================================================

def bubble_sort(arr):
    """Sort a list using bubble sort algorithm."""
    a = list(arr)
    n = len(a)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
                swapped = True
        if not swapped:
            break
    return a


def quick_sort(arr):
    """Sort a list using quick sort algorithm."""
    if len(arr) <= 1:
        return list(arr)
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)


# ============================================================
# Main Entry Point
# ============================================================

def main():
    print("=" * 60)
    print("Application Demo")
    print("=" * 60)

    print("\n--- Greeting ---")
    print(greet("World"))

    print("\n--- Factorial ---")
    for num in [0, 1, 5, 10]:
        print(f"  {num}! = {compute_factorial(num)}")

    print("\n--- Fibonacci ---")
    fib_values = [fibonacci(i) for i in range(15)]
    print(f"  First 15: {fib_values}")

    print("\n--- Primes up to 50 ---")
    print(f"  {generate_primes(50)}")

    print("\n--- Hash Example ---")
    sample_text = "hello world"
    print(f"  SHA256 of '{sample_text}': {hash_string(sample_text)}")

    print("\n--- Flatten ---")
    nested = [1, [2, 3], [4, [5, 6]], 7]
    print(f"  {nested} -> {flatten_list(nested)}")

    print("\n--- Statistics ---")
    data = [random.randint(1, 100) for _ in range(20)]
    stats = Statistics(data)
    print(f"  Data: {data}")
    print(f"  Summary: {json.dumps(stats.summary(), indent=4)}")

    print("\n--- Task Manager ---")
    tm = TaskManager()
    tm.add_task("Write code", "Implement the main module", priority=2)
    tm.add_task("Write tests", "Cover edge cases", priority=1)
    tm.add_task("Deploy", "Push to production", priority=3)
    tm.complete_task(1)
    print(f"  Pending tasks: {len(tm.get_pending())}")
    print(f"  Completed tasks: {len(tm.get_completed())}")

    print("\n--- Sorting ---")
    unsorted = [random.randint(1, 50) for _ in range(10)]
    print(f"  Original:    {unsorted}")
    print(f"  Bubble sort: {bubble_sort(unsorted)}")
    print(f"  Quick sort:  {quick_sort(unsorted)}")

    print("\n" + "=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()