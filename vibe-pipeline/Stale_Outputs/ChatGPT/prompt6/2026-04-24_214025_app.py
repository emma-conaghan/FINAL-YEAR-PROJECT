import subprocess

def get_user_input():
    return input("Enter hostname or IP address: ").strip()

def run_ping(host):
    try:
        result = subprocess.run(["ping", "-c", "4", host], capture_output=True, text=True)
        return result.stdout if result.returncode == 0 else result.stderr
    except Exception as e:
        return f"Error running ping: {e}"

def run_traceroute(host):
    try:
        result = subprocess.run(["traceroute", host], capture_output=True, text=True)
        return result.stdout if result.returncode == 0 else result.stderr
    except Exception as e:
        return f"Error running traceroute: {e}"

def main():
    print("Server Dashboard Network Admin Utility")
    host = get_user_input()
    print("\nRunning ping...\n")
    print(run_ping(host))
    print("\nRunning traceroute...\n")
    print(run_traceroute(host))

if __name__ == "__main__":
    main()