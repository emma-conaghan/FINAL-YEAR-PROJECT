import json
import os

class UserProfile:
    def __init__(self, username, email, age, preferences=None):
        self.username = username
        self.email = email
        self.age = age
        self.preferences = preferences if preferences else {}

    def to_dict(self):
        return {
            "username": self.username,
            "email": self.email,
            "age": self.age,
            "preferences": self.preferences
        }

    @staticmethod
    def from_dict(data):
        return UserProfile(
            username=data.get("username", ""),
            email=data.get("email", ""),
            age=data.get("age", 0),
            preferences=data.get("preferences", {})
        )

def export_profile(profile, filename):
    with open(filename, 'w') as f:
        json.dump(profile.to_dict(), f, indent=4)

def import_profile(filename):
    if not os.path.exists(filename):
        print("File does not exist.")
        return None
    with open(filename, 'r') as f:
        data = json.load(f)
        return UserProfile.from_dict(data)

def main():
    profile = UserProfile("johndoe", "john@example.com", 30, {"theme": "dark", "notifications": True})

    print("Exporting profile to userprofile.json")
    export_profile(profile, "userprofile.json")

    print("Importing profile from userprofile.json")
    loaded_profile = import_profile("userprofile.json")

    if loaded_profile:
        print("Loaded Profile:")
        print(f"Username: {loaded_profile.username}")
        print(f"Email: {loaded_profile.email}")
        print(f"Age: {loaded_profile.age}")
        print(f"Preferences: {loaded_profile.preferences}")

if __name__ == "__main__":
    main()