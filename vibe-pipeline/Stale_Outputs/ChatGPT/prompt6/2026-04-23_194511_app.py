import subprocess

def main():
    print("Server Dashboard Admin Utility")
    print("Enter 'exit' to quit.")

    while True:
        target = input("Enter hostname or IP address: ").strip()
        if target.lower() == 'exit':
            break
        if not target:
            print("Please enter a valid hostname or IP address.")
            continue

        print("Select diagnostic check:")
        print("1. Ping")
        print("2. Traceroute")
        choice = input("Enter choice (1 or 2): ").strip()

        if choice == '1':
            run_ping(target)
        elif choice == '2':
            run_traceroute(target)
        else:
            print("Invalid choice, please select 1 or 2.")

def run_ping(target):
    print(f"Pinging {target}...")
    try:
        # For Windows replace '-c' with '-n', here assuming Unix-like system
        count_param = '-n' if subprocess.run(["uname"], capture_output=True, text=True).returncode else '-c'
    except Exception:
        count_param = '-c'

    try:
        result = subprocess.run(["ping", count_param, "4", target], capture_output=True, text=True, timeout=10)
        print(result.stdout)
        if result.returncode != 0:
            print("Ping command failed or host is unreachable.")
    except Exception as e:
        print(f"Error running ping: {e}")

def run_traceroute(target):
    print(f"Running traceroute to {target}...")
    try:
        # Windows traceroute is 'tracert', Unix-like is 'traceroute'
        import platform
        system = platform.system().lower()
        if system == 'windows':
            cmd = ["tracert", target]
        else:
            cmd = ["traceroute", target]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        print(result.stdout)
        if result.returncode != 0:
            print("Traceroute command failed or host is unreachable.")
    except Exception as e:
        print(f"Error running traceroute: {e}")

if __name__ == "__main__":
    main()