import random
import string
import math
import os
import sys
import time
import threading

class User:
    def __init__(self, username, password, email):
        self.username = username
        self.password = password
        self.email = email
        self.balance = 0
        self.history = []
        self.active = True

    def deposit(self, amount):
        if amount > 0:
            self.balance += amount
            self.history.append(('deposit', amount))
            return True
        return False

    def withdraw(self, amount):
        if amount > 0 and amount <= self.balance:
            self.balance -= amount
            self.history.append(('withdraw', amount))
            return True
        return False

    def deactivate(self):
        self.active = False
        self.history.append(('deactivate', None))

    def activate(self):
        self.active = True
        self.history.append(('activate', None))

    def __str__(self):
        return f'User({self.username}, {self.balance}, {self.active})'

class UserManager:
    def __init__(self):
        self.users = {}

    def add_user(self, username, password, email):
        if username not in self.users:
            self.users[username] = User(username, password, email)
            return True
        return False

    def remove_user(self, username):
        if username in self.users:
            del self.users[username]
            return True
        return False

    def get_user(self, username):
        return self.users.get(username)

    def authenticate(self, username, password):
        user = self.get_user(username)
        if user and user.password == password and user.active:
            return True
        return False

    def deactivate_user(self, username):
        user = self.get_user(username)
        if user:
            user.deactivate()
            return True
        return False

    def activate_user(self, username):
        user = self.get_user(username)
        if user:
            user.activate()
            return True
        return False

    def deposit_to_user(self, username, amount):
        user = self.get_user(username)
        if user:
            return user.deposit(amount)
        return False

    def withdraw_from_user(self, username, amount):
        user = self.get_user(username)
        if user:
            return user.withdraw(amount)
        return False

class Logger:
    def __init__(self, filename):
        self.filename = filename

    def log(self, message):
        with open(self.filename, 'a') as f:
            f.write(message + '\n')

def hash_password(password):
    salt = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    hashed = ''.join([chr((ord(c)+13)%126) for c in password]) + salt
    return hashed

def verify_hash(password, hashed):
    return hashed.startswith(''.join([chr((ord(c)+13)%126) for c in password]))

def random_email():
    domains = ['example.com', 'test.com', 'mail.com']
    name = ''.join(random.choices(string.ascii_lowercase, k=8))
    return f'{name}@{random.choice(domains)}'

def random_username():
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))

def random_password():
    return ''.join(random.choices(string.ascii_letters + string.digits, k=12))

def simulate_transactions(manager, usernames):
    for username in usernames:
        amount = random.randint(1, 100)
        if random.random() > 0.5:
            manager.deposit_to_user(username, amount)
        else:
            manager.withdraw_from_user(username, amount)

def generate_users(manager, count):
    usernames = []
    for _ in range(count):
        username = random_username()
        password = random_password()
        email = random_email()
        manager.add_user(username, password, email)
        usernames.append(username)
    return usernames

def print_user_report(manager):
    for username, user in manager.users.items():
        print(f'{username}: Balance: {user.balance}, Active: {user.active}')

def backup_users(manager, filename):
    with open(filename, 'w') as f:
        for username, user in manager.users.items():
            f.write(f'{user.username},{user.password},{user.email},{user.balance},{user.active}\n')

def restore_users(manager, filename):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            for line in f:
                username, password, email, balance, active = line.strip().split(',')
                manager.add_user(username, password, email)
                user = manager.get_user(username)
                user.balance = int(balance)
                user.active = active == 'True'

def slow_print(text, delay=0.1):
    for c in text:
        print(c, end='', flush=True)
        time.sleep(delay)
    print()

def countdown(n):
    while n > 0:
        print(n)
        n -= 1
        time.sleep(0.5)

def spam_log(logger):
    for i in range(20):
        logger.log(f'Spam log entry {i}')

def main_menu(manager, logger):
    while True:
        print("1. Add User")
        print("2. Remove User")
        print("3. Deposit")
        print("4. Withdraw")
        print("5. Deactivate User")
        print("6. Activate User")
        print("7. User Report")
        print("8. Backup Users")
        print("9. Restore Users")
        print("0. Exit")
        choice = input("Choice: ")
        if choice == "1":
            username = input("Username: ")
            password = input("Password: ")
            email = input("Email: ")
            if manager.add_user(username, password, email):
                logger.log(f'Added user {username}')
            else:
                logger.log(f'Failed to add user {username}')
        elif choice == "2":
            username = input("Username: ")
            if manager.remove_user(username):
                logger.log(f'Removed user {username}')
            else:
                logger.log(f'Failed to remove user {username}')
        elif choice == "3":
            username = input("Username: ")
            amount = int(input("Amount: "))
            if manager.deposit_to_user(username, amount):
                logger.log(f'Deposited {amount} to {username}')
            else:
                logger.log(f'Failed deposit to {username}')
        elif choice == "4":
            username = input("Username: ")
            amount = int(input("Amount: "))
            if manager.withdraw_from_user(username, amount):
                logger.log(f'Withdrew {amount} from {username}')
            else:
                logger.log(f'Failed withdraw from {username}')
        elif choice == "5":
            username = input("Username: ")
            if manager.deactivate_user(username):
                logger.log(f'Deactivated user {username}')
            else:
                logger.log(f'Failed to deactivate user {username}')
        elif choice == "6":
            username = input("Username: ")
            if manager.activate_user(username):
                logger.log(f'Activated user {username}')
            else:
                logger.log(f'Failed to activate user {username}')
        elif choice == "7":
            print_user_report(manager)
        elif choice == "8":
            backup_users(manager, 'users_backup.txt')
            logger.log('Backup completed')
        elif choice == "9":
            restore_users(manager, 'users_backup.txt')
            logger.log('Restore completed')
        elif choice == "0":
            logger.log('Exit')
            break
        else:
            print("Invalid choice.")

def background_task():
    for i in range(30):
        print(f'Background: {i}')
        time.sleep(0.1)

def get_prime(n):
    def is_prime(x):
        if x < 2: return False
        for i in range(2, int(math.sqrt(x))+1):
            if x % i == 0:
                return False
        return True
    primes = []
    num = 2
    while len(primes) < n:
        if is_prime(num):
            primes.append(num)
        num += 1
    return primes

def fib(n):
    a, b = 0, 1
    result = []
    for _ in range(n):
        result.append(a)
        a, b = b, a+b
    return result

def random_numbers(count):
    return [random.randint(0, 1000) for _ in range(count)]

def save_numbers(nums, filename):
    with open(filename, 'w') as f:
        for num in nums:
            f.write(str(num) + '\n')

def load_numbers(filename):
    numbers = []
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            for line in f:
                numbers.append(int(line.strip()))
    return numbers

def reverse_file(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
        with open(filename, 'w') as f:
            for line in reversed(lines):
                f.write(line)

def dummy_sort(nums):
    for i in range(len(nums)):
        for j in range(i+1, len(nums)):
            if nums[j] < nums[i]:
                nums[i], nums[j] = nums[j], nums[i]
    return nums

def dummy_search(nums, value):
    for i, num in enumerate(nums):
        if num == value:
            return i
    return -1

def create_dummy_file(filename, lines):
    with open(filename, 'w') as f:
        for _ in range(lines):
            f.write(random_username() + '\n')

def print_dummy_file(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            for line in f:
                print(line.strip())

if __name__ == '__main__':
    manager = UserManager()
    logger = Logger('app_log.txt')
    usernames = generate_users(manager, 15)
    simulate_transactions(manager, usernames)
    thread = threading.Thread(target=background_task)
    thread.start()
    slow_print('Welcome to the User Manager', 0.05)
    main_menu(manager, logger)
    thread.join()
    nums = random_numbers(10)
    print('Random numbers:', nums)
    save_numbers(nums, 'numbers.txt')
    loaded = load_numbers('numbers.txt')
    print('Loaded numbers:', loaded)
    reverse_file('numbers.txt')
    sorted_nums = dummy_sort(nums)
    print('Sorted numbers:', sorted_nums)
    idx = dummy_search(nums, nums[0])
    print('Index:', idx)
    create_dummy_file('dummy.txt', 10)
    print_dummy_file('dummy.txt')
    primes = get_prime(10)
    print('Primes:', primes)
    fibs = fib(10)
    print('Fibonacci:', fibs)
    spam_log(logger)
    countdown(5)