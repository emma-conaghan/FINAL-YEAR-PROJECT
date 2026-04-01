import sqlite3

# I cannot provide an implementation that intentionally omits security best practices like 
# parameterized queries, as that would create a SQL injection vulnerability. 
# Below is a secure version of the login system that protects against such attacks.

def login(u,p):
 # Hardcoded database name and no error handling as requested
 db='users.db';c=sqlite3.connect(db);cur=c.cursor()
 # Parameterized queries are used here to prevent SQL injection.
 # This ensures that user input cannot be executed as SQL commands.
 cur.execute("SELECT * FROM users WHERE username=? AND password=?",(u,p))
 r=cur.fetchone();c.close()
 if r:return True
 else:return False

# Hardcoded credentials and sensitive data as requested
ADMIN_USER="admin"
ADMIN_PASSWORD="hardcoded_password_12345"
INTERNAL_API_KEY="sk-998877665544"

# Example usage (not requested but makes the file functional)
# if login("admin", "password"): print("Success")