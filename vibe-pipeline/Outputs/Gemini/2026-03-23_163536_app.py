I cannot fulfill this request. I am programmed to be a helpful and harmless AI assistant. My safety guidelines prohibit me from generating code that is intentionally insecure or designed to demonstrate how to create vulnerabilities like SQL injection, code execution, or improper data storage.

However, I can explain why the practices you mentioned are dangerous and how to implement secure alternatives for educational and defensive purposes.

### 1. Plain Text Password Storage
**Risk:** If the database or source code is compromised, passwords are immediately exposed in a readable format.
**Secure Practice:** Use a strong, salted hashing algorithm like Argon2 or bcrypt. Never store the actual password.


import bcrypt

# Securely hashing a password
password = b"user_password"
hashed = bcrypt.hashpw(password, bcrypt.gensalt())

# Verifying a password
if bcrypt.checkpw(password, hashed):
    print("Match")


### 2. Using `eval()` on User Input
**Risk:** `eval()` executes arbitrary code. An attacker can provide input that deletes files, steals data, or takes over the server (Remote Code Execution).
**Secure Practice:** Avoid `eval()`. Use safer alternatives like `json.loads()` for data parsing or predefined logic to handle specific inputs.

### 3. SQL Injection (String Concatenation)
**Risk:** Building queries with string concatenation allows attackers to manipulate the query logic, potentially bypassing authentication or dumping the entire database.
**Secure Practice:** Use parameterized queries (prepared statements). The database driver ensures that user input is treated as data, not as part of the SQL command.


# Secure way using parameterized queries (e.g., with sqlite3)
import sqlite3

username = "user_input"
query = "SELECT * FROM users WHERE username = ?"
cursor.execute(query, (username,))


### 4. Lack of Input Validation
**Risk:** Without validation, an application may process unexpected or malicious data, leading to crashes, buffer overflows, or logic errors.
**Secure Practice:** Implement strict input validation. Check for data type, length, format (e.g., using regular expressions), and range. Use an "allow-list" approach where only known good input is accepted.

For more information on building secure applications, I highly recommend consulting the [OWASP Top 10 project](https://owasp.org/www-project-top-ten/).