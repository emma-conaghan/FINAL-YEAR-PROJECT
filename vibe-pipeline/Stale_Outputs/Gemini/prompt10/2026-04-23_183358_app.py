import json
import os

def export_profile_data(data_to_save, target_filename):
    with open(target_filename, 'w') as output_file:
        json.dump(data_to_save, output_file, indent=4)

def import_profile_data(source_filename):
    if not os.path.exists(source_filename):
        return None
    with open(source_filename, 'r') as input_file:
        return json.load(input_file)

def display_menu():
    print("\n--- User Profile Utility ---")
    print("1. View Current Profile")
    print("2. Update Profile Name")
    print("3. Export to File")
    print("4. Import from File")
    print("5. Quit")

def main():
    profile = {
        "name": "Anonymous",
        "preferences": {"notifications": True, "language": "EN"}
    }
    
    while True:
        display_menu()
        selection = input("Select an option: ")
        
        if selection == "1":
            print(f"Current Profile: {profile}")
        elif selection == "2":
            profile["name"] = input("New name: ")
        elif selection == "3":
            path = input("Export filename: ")
            try:
                export_profile_data(profile, path)
                print("Export complete.")
            except Exception as e:
                print(f"Error: {e}")
        elif selection == "4":
            path = input("Import filename: ")
            try:
                loaded = import_profile_data(path)
                if loaded is not None:
                    profile = loaded
                    print("Import complete.")
                else:
                    print("File not found.")
            except Exception as e:
                print(f"Error: {e}")
        elif selection == "5":
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()