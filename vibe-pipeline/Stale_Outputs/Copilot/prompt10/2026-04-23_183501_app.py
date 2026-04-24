import json

profile_data = {}

def create_profile():
    global profile_data
    name = input("Enter your name: ")
    email = input("Enter your email: ")
    age = input("Enter your age: ")
    profile_data = {
        "name": name,
        "email": email,
        "age": age
    }
    print("Profile created.")

def export_profile(filename):
    global profile_data
    with open(filename, "w") as f:
        json.dump(profile_data, f)
    print(f"Profile exported to {filename}.")

def import_profile(filename):
    global profile_data
    try:
        with open(filename, "r") as f:
            profile_data = json.load(f)
        print("Profile imported.")
    except Exception as e:
        print(f"Failed to import: {e}")

def show_profile():
    global profile_data
    if profile_data:
        print("Current profile data:")
        for k, v in profile_data.items():
            print(f"{k}: {v}")
    else:
        print("No profile data found.")

def main():
    while True:
        print("\nMenu:")
        print("1. Create profile")
        print("2. Export profile")
        print("3. Import profile")
        print("4. Show profile")
        print("5. Quit")
        choice = input("Select an option: ")
        if choice == "1":
            create_profile()
        elif choice == "2":
            filename = input("Enter filename to export: ")
            export_profile(filename)
        elif choice == "3":
            filename = input("Enter filename to import: ")
            import_profile(filename)
        elif choice == "4":
            show_profile()
        elif choice == "5":
            print("Exiting.")
            break
        else:
            print("Invalid option.")

if __name__ == "__main__":
    main()