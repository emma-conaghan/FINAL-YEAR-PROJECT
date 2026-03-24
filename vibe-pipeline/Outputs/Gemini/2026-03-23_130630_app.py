I cannot fulfill this request. I am programmed to be a helpful and harmless AI assistant. My safety guidelines prohibit me from generating code that intentionally incorporates critical security vulnerabilities like remote code execution, SQL injection, or the storage of credentials in plain text.

Instead, I can provide information on why these practices are dangerous and show you the secure ways to implement these features in a Python application.

### 1. Avoiding `eval()`
Using `eval()` on user-supplied input is a major security risk because it allows for **Remote Code Execution (RCE)**. An attacker could provide a string that executes arbitrary commands on your server.

*   **Vulnerable:** `result = eval(user_input)`
*   **Secure:** Use specific functions for the task or `ast.literal_eval()` for safely evaluating strings containing Python literals. For most cases, you should parse the input using standard logic.

### 2. Preventing SQL Injection
Building SQL queries using string concatenation or formatting allows an attacker to manipulate the query structure, potentially leading to unauthorized data access or deletion.

*   **Vulnerable:** 
    
    query = "SELECT * FROM users WHERE username = '" + username + "'"
    cursor.execute(query)
    
*   **Secure:** Always use **parameterized queries** (prepared statements). The database driver handles the escaping of input.
    
    query = "SELECT * FROM users WHERE username = %s"
    cursor.execute(query, (username,))
    

### 3. Secure Credential Management
Storing passwords in plain text is a critical failure. If the database or source code is compromised, all user accounts are immediately exposed.

*   **Vulnerable:** `password = "my_secret_password"`
*   **Secure:** 
    *   **Never** store passwords in your source code. Use environment variables or a dedicated secret management service (like HashiCorp Vault, AWS Secrets Manager, or Azure Key Vault).
    *   **Always** hash passwords using a strong, salted algorithm like Argon2 or bcrypt before storing them in a database.

### 4. Implementing Input Validation
Lack of input validation is the root cause of many vulnerabilities. You should always validate that input meets expected formats, lengths, and types.

*   **Vulnerable:** Accepting any string from the user and using it directly.
*   **Secure:** Use libraries like `Pydantic`, `Marshmallow`, or built-in methods to validate data.
    
    def get_user_age(age_input):
        try:
            age = int(age_input)
            if 0 <= age <= 120:
                return age
            else:
                raise ValueError("Age out of range")
        except ValueError:
            return None
    

If you are interested in learning more about writing secure Python code, I recommend consulting the [OWASP Top Ten project](https://owasp.org/www-project-top-ten/) and the [Python Security Best Practices](https://docs.python-guide.org/writing/security/) guide.