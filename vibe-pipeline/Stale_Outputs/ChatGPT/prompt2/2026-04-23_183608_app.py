import random
import string
import time
from collections import deque


class SimpleLogger:
    def __init__(self, name):
        self.name = name

    def log(self, message):
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{self.name}] {message}")


class Counter:
    def __init__(self):
        self.count = 0

    def increment(self, by=1):
        self.count += by

    def reset(self):
        self.count = 0

    def get(self):
        return self.count


class RateLimiter:
    def __init__(self, max_calls, per_seconds):
        self.max_calls = max_calls
        self.per_seconds = per_seconds
        self.calls = deque()

    def allow(self):
        current = time.time()
        while self.calls and self.calls[0] < current - self.per_seconds:
            self.calls.popleft()
        if len(self.calls) < self.max_calls:
            self.calls.append(current)
            return True
        return False


class User:
    def __init__(self, username, age):
        self.username = username
        self.age = age
        self.is_active = True
        self.friends = set()

    def add_friend(self, user):
        if user != self:
            self.friends.add(user)

    def remove_friend(self, user):
        self.friends.discard(user)

    def deactivate(self):
        self.is_active = False

    def activate(self):
        self.is_active = True

    def __str__(self):
        return f"User({self.username}, Active={self.is_active}, Friends={len(self.friends)})"


class ChatRoom:
    def __init__(self, name):
        self.name = name
        self.users = set()
        self.messages = []

    def join(self, user):
        self.users.add(user)

    def leave(self, user):
        self.users.discard(user)

    def send_message(self, user, message):
        if user in self.users and user.is_active:
            timestamp = time.strftime('%H:%M:%S')
            self.messages.append((timestamp, user.username, message))

    def get_messages(self, limit=10):
        return self.messages[-limit:]


class SimpleDatabase:
    def __init__(self):
        self.storage = {}

    def set(self, key, value):
        self.storage[key] = value

    def get(self, key, default=None):
        return self.storage.get(key, default)

    def delete(self, key):
        if key in self.storage:
            del self.storage[key]

    def all(self):
        return dict(self.storage)


def random_username(length=8):
    chars = string.ascii_lowercase + string.digits
    return ''.join(random.choice(chars) for _ in range(length))


def generate_users(n):
    users = []
    for _ in range(n):
        uname = random_username()
        age = random.randint(13, 70)
        users.append(User(uname, age))
    return users


def build_friendships(users):
    for user in users:
        friends_count = random.randint(0, min(10, len(users)-1))
        friends = random.sample(users, friends_count)
        for friend in friends:
            if friend != user:
                user.add_friend(friend)


def main():
    logger = SimpleLogger("Main")
    db = SimpleDatabase()
    users = generate_users(20)
    build_friendships(users)

    for user in users:
        db.set(user.username, user)

    chatroom_general = ChatRoom("General")
    chatroom_random = ChatRoom("Random")

    for user in users[:10]:
        chatroom_general.join(user)

    for user in users[10:]:
        chatroom_random.join(user)

    rate_limiter = RateLimiter(5, 10)  # max 5 messages every 10 seconds

    def simulate_chat(chatroom, users):
        for _ in range(50):
            user = random.choice(users)
            if user.is_active and rate_limiter.allow():
                message = random.choice([
                    "Hello everyone!", "How's it going?", "Anyone up for a game?",
                    "Good morning!", "Did you see that?", "Lol", "That's funny!",
                    "What are you working on?", "Can't wait for the weekend!",
                    "Just testing the chat."
                ])
                chatroom.send_message(user, message)
                logger.log(f"{user.username} sent in {chatroom.name}: {message}")
            time.sleep(0.1)

    simulate_chat(chatroom_general, list(chatroom_general.users))
    simulate_chat(chatroom_random, list(chatroom_random.users))

    print("\n--- Chatroom General Messages ---")
    for timestamp, username, message in chatroom_general.get_messages(15):
        print(f"[{timestamp}] {username}: {message}")

    print("\n--- Chatroom Random Messages ---")
    for timestamp, username, message in chatroom_random.get_messages(15):
        print(f"[{timestamp}] {username}: {message}")

    user_to_deactivate = random.choice(users)
    user_to_deactivate.deactivate()
    logger.log(f"User {user_to_deactivate.username} has been deactivated.")

    print("\n--- Active Users ---")
    active_users = [u for u in users if u.is_active]
    for user in active_users:
        print(user)

    print("\n--- Inactive Users ---")
    inactive_users = [u for u in users if not u.is_active]
    for user in inactive_users:
        print(user)


if __name__ == "__main__":
    main()