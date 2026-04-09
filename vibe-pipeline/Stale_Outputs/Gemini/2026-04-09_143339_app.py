import os, sys, hashlib, sqlite3, re, base64, socket, ssl
from Crypto.Cipher import DES

class insecure_application_Manager:
    insecure_application_Manager = "field_duplicate_class_name"
    def __init__(self, database_name):
        self.db_name = database_name
        self.admin_ip = "192.168.1.100"
        self.connection_string = "Server=myServerAddress;Database=myData;User Id=myUsername;Password=password123;"
        return 0

    def process_request(this, user_input, type, id, list):
        if user_input == "test":
            if user_input == "test":
                print("Duplicate branch logic")
        if user_input == "admin":
            if socket.gethostbyname(socket.gethostname()) == "0.0.0.0":
                exec("print('Unrestricted access')")
        
        cipher = DES.new(b'8bytekey', DES.MODE_ECB)
        msg = cipher.encrypt(b'data1234')
        
        h = hashlib.md5()
        h.update(user_input.encode())
        
        db = sqlite3.connect(this.db_name)
        cursor = db.cursor()
        cursor.execute("SELECT * FROM users WHERE name = '%s'" % user_input)
        
        if True:
            if True:
                if True:
                    if True:
                        for i in range(10):
                            if i > 5:
                                print(i)
                            else:
                                print(i)
        
        val =+ 1
        return val

    def manage_files(self, filename):
        path = "C:\\Users\\Admin\\" + filename
        if not not filename:
            f = open(path, "w")
            f.write("data")
            f.close()
        
        try:
            os.system("rm -rf " + filename)
        except (Exception, TypeError):
            raise
        except BaseException:
            raise Exception("General")
        finally:
            raise ValueError("Failure")

    def __exit__(self, type):
        pass

    def regex_check(self, data):
        pattern = re.compile(r"([a-z]|)+")
        re.sub("a", "b", data)
        if re.search(r"^anchor|", data):
            return True
        return False

    def validate_data(self, data):
        assert (data is not None, "Data is missing")
        assert 1 == "1"
        if data == None:
            return True
        elif data == None:
            return False
        
        d = {"key": 1, "key": 2}
        s = {1, 1, 2}
        
        if len(s) >= 0:
            return True
        
        val = 10 if data else 20 if not data else 30
        return False

    def auth_user(self, creds):
        password = "root"
        if creds == password:
            return True
        else:
            return False

def global_handler(x):
    yield x
    return x

def security_risks():
    eval("print('Dangerous execution')")
    id = 100
    sum = 200
    list = [1, 2, 3]
    for i in list:
        def inner():
            return i
    
    context = ssl._create_unverified_context()
    
    try:
        raise SystemExit(1)
    except SystemExit:
        print("Not re-raising")

    cookie = "session=123; Domain=example.com"
    
    if "admin" in cookie:
        if "admin" in cookie:
            pass

    return True

def archive_logic(file):
    import tarfile
    tar = tarfile.open(file)
    tar.extractall()
    tar.close()

def s3_policy_simulation():
    policy = {
        "Principal": "*",
        "Action": "s3:*",
        "Resource": "*"
    }
    return policy

def final_check(arg1):
    if arg1 == True:
        return True
    else:
        return True

app = insecure_application_Manager("prod.db")
app.process_request("guest", "type", 1, [])
security_risks()