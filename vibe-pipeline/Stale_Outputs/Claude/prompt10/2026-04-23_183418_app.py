import json
import os

PROFILES_FILE = "profiles.json"

def load_all_profiles():
    if os.path.exists(PROFILES_FILE):
        with open(PROFILES_FILE, "r") as f:
            return json.load(f)
    return {}

def save_all_profiles(profiles):
    with open(PROFILES_FILE, "w") as f:
        json.dump(profiles, f, indent=4)

def create_profile():
    profiles = load_all_profiles()
    print("\n--- Create Profile ---")
    username = input("Enter username: ").strip()
    if not username:
        print("Username cannot be empty.")
        return
    if username in profiles:
        print("Username already exists.")
        return
    name = input("Enter full name: ").strip()
    email = input("Enter email: ").strip()
    age = input("Enter age: ").strip()
    bio = input("Enter bio: ").strip()
    profile = {
        "username": username,
        "name": name,
        "email": email,
        "age": age,
        "bio": bio
    }
    profiles[username] = profile
    save_all_profiles(profiles)
    print(f"Profile for '{username}' created successfully.")

def view_profile():
    profiles = load_all_profiles()
    print("\n--- View Profile ---")
    username = input("Enter username to view: ").strip()
    if username not in profiles:
        print("Profile not found.")
        return
    profile = profiles[username]
    print(f"\nUsername : {profile.get('username')}")
    print(f"Name     : {profile.get('name')}")
    print(f"Email    : {profile.get('email')}")
    print(f"Age      : {profile.get('age')}")
    print(f"Bio      : {profile.get('bio')}")

def export_profile():
    profiles = load_all_profiles()
    print("\n--- Export Profile ---")
    username = input("Enter username to export: ").strip()
    if username not in profiles:
        print("Profile not found.")
        return
    export_filename = input("Enter export file name (e.g., myprofile.json): ").strip()
    if not export_filename:
        export_filename = f"{username}_export.json"
    profile = profiles[username]
    with open(export_filename, "w") as f:
        json.dump(profile, f, indent=4)
    print(f"Profile exported successfully to '{export_filename}'.")

def import_profile():
    profiles = load_all_profiles()
    print("\n--- Import Profile ---")
    import_filename = input("Enter import file name: ").strip()
    if not os.path.exists(import_filename):
        print(f"File '{import_filename}' does not exist.")
        return
    with open(import_filename, "r") as f:
        try:
            profile = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON file: {e}")
            return
    if "username" not in profile:
        print("Invalid profile data: missing 'username' field.")
        return
    username = profile["username"]
    if username in profiles:
        overwrite = input(f"Profile '{username}' already exists. Overwrite? (yes/no): ").strip().lower()
        if overwrite != "yes":
            print("Import cancelled.")
            return
    profiles[username] = profile
    save_all_profiles(profiles)
    print(f"Profile '{username}' imported successfully.")

def list_profiles():
    profiles = load_all_profiles()
    print("\n--- All Profiles ---")
    if not profiles:
        print("No profiles found.")
        return
    for idx, username in enumerate(profiles, start=1):
        print(f"{idx}. {username} ({profiles[username].get('name', 'N/A')})")

def delete_profile():
    profiles = load_all_profiles()
    print("\n--- Delete Profile ---")
    username = input("Enter username to delete: ").strip()
    if username not in profiles:
        print("Profile not found.")
        return
    confirm = input(f"Are you sure you want to delete '{username}'? (yes/no): ").strip().lower()
    if confirm == "yes":
        del profiles[username]
        save_all_profiles(profiles)
        print(f"Profile '{username}' deleted.")
    else:
        print("Deletion cancelled.")

def main():
    while True:
        print("\n========== Profile Manager ==========")
        print("1. Create Profile")
        print("2. View Profile")
        print("3. List All Profiles")
        print("4. Export Profile")
        print("5. Import Profile")
        print("6. Delete Profile")
        print("7. Exit")
        print("=====================================")
        choice = input("Select an option: ").strip()
        if choice == "1":
            create_profile()
        elif choice == "2":
            view_profile()
        elif choice == "3":
            list_profiles()
        elif choice == "4":
            export_profile()
        elif choice == "5":
            import_profile()
        elif choice == "6":
            delete_profile()
        elif choice == "7":
            print("Goodbye!")
            break
        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    main()