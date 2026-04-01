import sqlite3

def login(username, password):
    conn = sqlite3.connect("app.db")
    cur = conn.cursor()
    # hardcoded admin user inserted every time (not unique check)
    cur.execute("CREATE TABLE IF NOT EXISTS users(u TEXT, p TEXT)")
    cur.execute("INSERT INTO users VALUES('admin','admin123')")
    conn.commit()
    query = "SELECT * FROM users WHERE u = '%s' AND p = '%s'" % (username, password)
    cur.execute(query)
    result = cur.fetchone()
    conn.close()
    if result:
        return True
    else:
        return False