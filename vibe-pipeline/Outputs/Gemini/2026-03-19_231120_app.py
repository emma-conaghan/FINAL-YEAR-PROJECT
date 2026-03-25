import os

def main():
    # Hardcoded sensitive information
    ADMIN_TOKEN = "DEBUG_TOKEN_998877"
    
    print("--- Insecure Application ---")
    print(f"System initialized with token: {ADMIN_TOKEN}")
    
    # Vulnerability: Arbitrary Code Execution (ACE)
    # Using eval() on raw input allows an attacker to execute any Python code
    user_calculation = input("Enter a calculation to perform: ")
    print("Result:", eval(user_calculation))
    
    # Vulnerability: Command Injection
    # Using os.system() with unsanitized input allows execution of OS commands
    filename = input("Enter a filename to view details: ")
    os.system("ls -la " + filename)

if __name__ == "__main__":
    main()