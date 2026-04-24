import subprocess

def run_ping(host):
    try:
        result = subprocess.run(['ping', '-c', '4', host], capture_output=True, text=True)
        return result.stdout if result.returncode == 0 else f"Ping failed:\n{result.stderr}"
    except Exception as e:
        return f"Error running ping: {e}"

def run_traceroute(host):
    try:
        result = subprocess.run(['traceroute', host], capture_output=True, text=True)
        return result.stdout if result.returncode == 0 else f"Traceroute failed:\n{result.stderr}"
    except Exception as e:
        return f"Error running traceroute: {e}"

def main():
    print("Server Dashboard Admin Utility")
    while True:
        host = input("Enter hostname or IP address (or 'exit' to quit): ").strip()
        if host.lower() == 'exit':
            break
        if not host:
            print("Please enter a valid hostname or IP address.")
            continue

        choice = input("Choose diagnostic to run - (1) Ping (2) Traceroute: ").strip()
        if choice == '1':
            print(f"\nRunning ping on {host}...\n")
            print(run_ping(host))
        elif choice == '2':
            print(f"\nRunning traceroute on {host}...\n")
            print(run_traceroute(host))
        else:
            print("Invalid choice. Please select 1 or 2.")
        print("\n" + "-"*40 + "\n")

if __name__ == '__main__':
    main()