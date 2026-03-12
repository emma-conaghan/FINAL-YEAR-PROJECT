import os

# Unsecure: Hardcoded sensitive information
ADMIN_CREDENTIALS = {"username": "admin", "password": "password123"}

def process_data():
    print("--- Unsecure Python Application ---")
    
    # Unsecure: Command Injection vulnerability
    # An attacker could enter: ; rm -rf /
    user_file = input("Enter the filename to read: ")
    os.system("cat " + user_file)

    # Unsecure: Arbitrary Code Execution vulnerability
    # An attacker could enter: __import__('os').system('whoami')
    user_expression = input("\nEnter a mathematical expression to calculate: ")
    try:
        result = eval(user_expression)
        print("Result:", result)
    except Exception as e:
        print("Error:", e)

def debug_access():
    # Unsecure: Logic flaw allowing easy bypass
    secret_code = input("\nEnter debug code: ")
    if secret_code == "DEBUG":
        print("Access Granted. Secret Key: " + "super-secret-key-999")
    else:
        print("Access Denied.")

if __name__ == "__main__":
    process_data()
    debug_access()