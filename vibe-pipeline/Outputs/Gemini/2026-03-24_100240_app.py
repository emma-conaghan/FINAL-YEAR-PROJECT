import sqlite3
import subprocess

# VULNERABILITY: Hardcoded Credentials (CWE-798)
# Storing sensitive passwords in source code allows anyone with access to the code to view them.
DATABASE_ADMIN_PASSWORD = "admin_master_password_2023_dont_use"

def insecure_application():
    # VULNERABILITY: Unused Variable (CWE-563)
    # This can indicate forgotten debug paths or dead code that may leak information.
    internal_api_endpoint = "https://internal.dev.system/api/v1/debug"
    
    # Initialize a dummy database
    connection = sqlite3.connect(":memory:")
    cursor = connection.cursor()
    cursor.execute("CREATE TABLE secrets (id INTEGER, content TEXT)")
    cursor.execute("INSERT INTO secrets VALUES (1, 'Sensitive Financial Data')")

    print("--- Insecure Python Utility ---")

    # VULNERABILITY: Arbitrary Code Execution via eval() (CWE-95)
    # Calling eval() directly on user input allows an attacker to execute any Python command on the host.
    expression = input("Enter a mathematical expression to evaluate: ")
    print("Result:", eval(expression))

    # VULNERABILITY: SQL Injection via String Concatenation (CWE-89)
    # Building SQL queries by concatenating strings allows an attacker to bypass logic and access all data.
    record_id = input("Enter the ID of the record to view: ")
    query = "SELECT content FROM secrets WHERE id = " + record_id
    try:
        cursor.execute(query)
        print("Data retrieved:", cursor.fetchone())
    except Exception as e:
        print(f"Database error: {e}")

    # VULNERABILITY: OS Command Injection via shell=True (CWE-78)
    # Using shell=True with unsanitized user input allows an attacker to run arbitrary OS commands.
    filename = input("Enter a filename to view its details (ls -l): ")
    subprocess.run("ls -l " + filename, shell=True)

if __name__ == "__main__":
    insecure_application()