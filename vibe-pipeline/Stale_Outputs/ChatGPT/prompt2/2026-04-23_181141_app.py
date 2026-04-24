import random
import string
import time
import math
import os

class User:
    def __init__(self, username, age, email):
        self.username = username
        self.age = age
        self.email = email
        self.logged_in = False

    def login(self, password):
        if password == "password123":
            self.logged_in = True
            return True
        return False

    def logout(self):
        self.logged_in = False

    def __str__(self):
        return f"User({self.username}, {self.age}, {self.email}, logged_in={self.logged_in})"

class AuthSystem:
    def __init__(self):
        self.users = {}

    def register(self, username, age, email, password):
        if username in self.users:
            return False
        self.users[username] = {
            "age": age,
            "email": email,
            "password": password
        }
        return True

    def authenticate(self, username, password):
        user = self.users.get(username)
        if user and user["password"] == password:
            return True
        return False

    def change_password(self, username, old_password, new_password):
        user = self.users.get(username)
        if user and user["password"] == old_password:
            user["password"] = new_password
            return True
        return False

class Calculator:
    @staticmethod
    def add(a, b):
        return a + b

    @staticmethod
    def subtract(a, b):
        return a - b

    @staticmethod
    def multiply(a, b):
        return a * b

    @staticmethod
    def divide(a, b):
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b

    @staticmethod
    def power(a, b):
        return a ** b

    @staticmethod
    def factorial(n):
        if n < 0:
            raise ValueError("Factorial not defined for negative values")
        return math.factorial(n)

class RandomUtils:
    @staticmethod
    def random_int(min_val, max_val):
        return random.randint(min_val, max_val)

    @staticmethod
    def random_choice(seq):
        if not seq:
            return None
        return random.choice(seq)

    @staticmethod
    def random_string(length):
        letters = string.ascii_letters + string.digits
        return ''.join(random.choice(letters) for _ in range(length))

    @staticmethod
    def shuffle_list(lst):
        random.shuffle(lst)
        return lst

class FileHandler:
    @staticmethod
    def write_file(filename, content):
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)

    @staticmethod
    def read_file(filename):
        if not os.path.exists(filename):
            return None
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def append_file(filename, content):
        with open(filename, "a", encoding="utf-8") as f:
            f.write(content)

    @staticmethod
    def delete_file(filename):
        if os.path.exists(filename):
            os.remove(filename)
            return True
        return False

class Timer:
    def __init__(self):
        self.start_time = None
        self.running = False

    def start(self):
        if not self.running:
            self.start_time = time.time()
            self.running = True

    def stop(self):
        if self.running:
            elapsed = time.time() - self.start_time
            self.running = False
            self.start_time = None
            return elapsed
        return None

    def reset(self):
        self.start_time = None
        self.running = False

class TodoList:
    def __init__(self):
        self.tasks = []

    def add_task(self, task):
        if task and task not in self.tasks:
            self.tasks.append(task)
            return True
        return False

    def remove_task(self, task):
        if task in self.tasks:
            self.tasks.remove(task)
            return True
        return False

    def get_tasks(self):
        return self.tasks[:]

    def clear_tasks(self):
        self.tasks.clear()

class Matrix:
    def __init__(self, rows, cols, fill=0):
        self.rows = rows
        self.cols = cols
        self.data = [[fill for _ in range(cols)] for _ in range(rows)]

    def set(self, row, col, value):
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self.data[row][col] = value
            return True
        return False

    def get(self, row, col):
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.data[row][col]
        return None

    def add(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrix dimensions must match for addition")
        result = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[i][j] = self.data[i][j] + other.data[i][j]
        return result

    def multiply(self, other):
        if self.cols != other.rows:
            raise ValueError("Matrix dimensions incompatible for multiplication")
        result = Matrix(self.rows, other.cols)
        for i in range(self.rows):
            for j in range(other.cols):
                sum_prod = 0
                for k in range(self.cols):
                    sum_prod += self.data[i][k] * other.data[k][j]
                result.data[i][j] = sum_prod
        return result

    def transpose(self):
        result = Matrix(self.cols, self.rows)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[j][i] = self.data[i][j]
        return result

    def __str__(self):
        return '\n'.join(['\t'.join(map(str, row)) for row in self.data])

def print_menu():
    print("=== Simple App Menu ===")
    print("1. Register user")
    print("2. Login user")
    print("3. Calculator operations")
    print("4. Generate random string")
    print("5. File operations")
    print("6. Timer")
    print("7. Todo List")
    print("8. Matrix operations")
    print("9. Exit")

def calculator_menu():
    print("=== Calculator ===")
    print("a. Add")
    print("b. Subtract")
    print("c. Multiply")
    print("d. Divide")
    print("e. Power")
    print("f. Factorial")
    print("g. Back")

def file_menu():
    print("=== File Operations ===")
    print("a. Write file")
    print("b. Read file")
    print("c. Append file")
    print("d. Delete file")
    print("e. Back")

def todo_menu():
    print("=== Todo List ===")
    print("a. Add task")
    print("b. Remove task")
    print("c. View tasks")
    print("d. Clear tasks")
    print("e. Back")

def matrix_menu():
    print("=== Matrix Operations ===")
    print("a. Create matrix")
    print("b. Set value")
    print("c. Get value")
    print("d. Add matrices")
    print("e. Multiply matrices")
    print("f. Transpose matrix")
    print("g. Back")

def main():
    auth = AuthSystem()
    users = {}
    todo = TodoList()
    timer = Timer()
    matrices = {}

    current_user = None

    while True:
        print_menu()
        choice = input("Choose an option: ").strip()

        if choice == "1":
            username = input("Username: ")
            age = input("Age: ")
            email = input("Email: ")
            password = input("Password: ")
            if not age.isdigit():
                print("Invalid age")
                continue
            age = int(age)
            if auth.register(username, age, email, password):
                print(f"User {username} registered successfully.")
            else:
                print("Username already exists.")

        elif choice == "2":
            username = input("Username: ")
            password = input("Password: ")
            if auth.authenticate(username, password):
                current_user = User(username, auth.users[username]["age"], auth.users[username]["email"])
                current_user.login(password)
                print(f"User {username} logged in.")
            else:
                print("Failed to login.")

        elif choice == "3":
            if current_user is None or not current_user.logged_in:
                print("Please login first.")
                continue
            while True:
                calculator_menu()
                op = input("Operation: ").strip().lower()
                if op == "g":
                    break
                elif op in {"a", "b", "c", "d", "e"}:
                    try:
                        a = float(input("Enter first number: "))
                        b = float(input("Enter second number: "))
                        result = None
                        if op == "a":
                            result = Calculator.add(a, b)
                        elif op == "b":
                            result = Calculator.subtract(a, b)
                        elif op == "c":
                            result = Calculator.multiply(a, b)
                        elif op == "d":
                            result = Calculator.divide(a, b)
                        elif op == "e":
                            result = Calculator.power(a, b)
                        print(f"Result: {result}")
                    except Exception as ex:
                        print(f"Error: {ex}")
                elif op == "f":
                    try:
                        n = int(input("Enter number for factorial: "))
                        result = Calculator.factorial(n)
                        print(f"Result: {result}")
                    except Exception as ex:
                        print(f"Error: {ex}")
                else:
                    print("Invalid option.")

        elif choice == "4":
            length = input("Enter length of random string: ")
            if not length.isdigit():
                print("Invalid length.")
                continue
            length = int(length)
            rand_str = RandomUtils.random_string(length)
            print(f"Random string: {rand_str}")

        elif choice == "5":
            while True:
                file_menu()
                op = input("Choose file operation: ").strip().lower()
                if op == "e":
                    break
                elif op == "a":
                    filename = input("Filename: ")
                    content = input("Content to write: ")
                    FileHandler.write_file(filename, content)
                    print("Content written.")
                elif op == "b":
                    filename = input("Filename: ")
                    content = FileHandler.read_file(filename)
                    if content is None:
                        print("File does not exist.")
                    else:
                        print("Content:")
                        print(content)
                elif op == "c":
                    filename = input("Filename: ")
                    content = input("Content to append: ")
                    FileHandler.append_file(filename, content)
                    print("Content appended.")
                elif op == "d":
                    filename = input("Filename: ")
                    if FileHandler.delete_file(filename):
                        print("File deleted.")
                    else:
                        print("File does not exist.")
                else:
                    print("Invalid option.")

        elif choice == "6":
            print("Timer commands: start, stop, reset, back")
            while True:
                cmd = input("Timer command: ").strip().lower()
                if cmd == "start":
                    timer.start()
                    print("Timer started.")
                elif cmd == "stop":
                    elapsed = timer.stop()
                    if elapsed is not None:
                        print(f"Elapsed time: {elapsed:.2f} seconds")
                    else:
                        print("Timer is not running.")
                elif cmd == "reset":
                    timer.reset()
                    print("Timer reset.")
                elif cmd == "back":
                    break
                else:
                    print("Invalid command.")

        elif choice == "7":
            while True:
                todo_menu()
                op = input("Choose task option: ").strip().lower()
                if op == "e":
                    break
                elif op == "a":
                    task = input("Enter task: ")
                    if todo.add_task(task):
                        print("Task added.")
                    else:
                        print("Task already exists or empty.")
                elif op == "b":
                    task = input("Enter task to remove: ")
                    if todo.remove_task(task):
                        print("Task removed.")
                    else:
                        print("Task not found.")
                elif op == "c":
                    tasks = todo.get_tasks()
                    if not tasks:
                        print("No tasks.")
                    else:
                        print("Tasks:")
                        for t in tasks:
                            print(f"- {t}")
                elif op == "d":
                    todo.clear_tasks()
                    print("Tasks cleared.")
                else:
                    print("Invalid option.")

        elif choice == "8":
            current_matrix = None
            other_matrix = None
            while True:
                matrix_menu()
                op = input("Matrix operation: ").strip().lower()
                if op == "g":
                    break
                elif op == "a":
                    name = input("Matrix name: ")
                    rows = input("Rows: ")
                    cols = input("Cols: ")
                    fill = input("Fill value (default 0): ")
                    if not rows.isdigit() or not cols.isdigit():
                        print("Invalid rows or cols.")
                        continue
                    rows, cols = int(rows), int(cols)
                    if fill.strip() == "":
                        fill = 0
                    else:
                        try:
                            fill = int(fill)
                        except:
                            print("Invalid fill value.")
                            continue
                    matrices[name] = Matrix(rows, cols, fill)
                    print(f"Matrix {name} created.")
                elif op == "b":
                    name = input("Matrix name: ")
                    if name not in matrices:
                        print("Matrix not found.")
                        continue
                    row = input("Row index: ")
                    col = input("Col index: ")
                    value = input("Value: ")
                    if not (row.isdigit() and col.isdigit()):
                        print("Invalid row or col index.")
                        continue
                    row, col = int(row), int(col)
                    try:
                        value = int(value)
                    except:
                        print("Invalid value.")
                        continue
                    if matrices[name].set(row, col, value):
                        print("Value set.")
                    else:
                        print("Set failed.")
                elif op == "c":
                    name = input("Matrix name: ")
                    if name not in matrices:
                        print("Matrix not found.")
                        continue
                    row = input("Row index: ")
                    col = input("Col index: ")
                    if not (row.isdigit() and col.isdigit()):
                        print("Invalid row or col index.")
                        continue
                    row, col = int(row), int(col)
                    val = matrices[name].get(row, col)
                    if val is None:
                        print("Get failed.")
                    else:
                        print(f"Value: {val}")
                elif op == "d":
                    name1 = input("First matrix name: ")
                    name2 = input("Second matrix name: ")
                    if name1 not in matrices or name2 not in matrices:
                        print("One or both matrices not found.")
                        continue
                    try:
                        result = matrices[name1].add(matrices[name2])
                        print("Result of addition:")
                        print(result)
                    except Exception as e:
                        print(f"Error: {e}")
                elif op == "e":
                    name1 = input("First matrix name: ")
                    name2 = input("Second matrix name: ")
                    if name1 not in matrices or name2 not in matrices:
                        print("One or both matrices not found.")
                        continue
                    try:
                        result = matrices[name1].multiply(matrices[name2])
                        print("Result of multiplication:")
                        print(result)
                    except Exception as e:
                        print(f"Error: {e}")
                elif op == "f":
                    name = input("Matrix name: ")
                    if name not in matrices:
                        print("Matrix not found.")
                        continue
                    transposed = matrices[name].transpose()
                    print("Transposed matrix:")
                    print(transposed)
                else:
                    print("Invalid option.")

        elif choice == "9":
            print("Exiting application.")
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()