import subprocess

def ping_host(host):
    try:
        result = subprocess.run(['ping', '-c', '4', host], capture_output=True, text=True, timeout=10)
        return result.stdout if result.returncode == 0 else f"Ping failed:\n{result.stderr}"
    except subprocess.TimeoutExpired:
        return "Ping command timed out."

def traceroute_host(host):
    try:
        result = subprocess.run(['traceroute', host], capture_output=True, text=True, timeout=30)
        return result.stdout if result.returncode == 0 else f"Traceroute failed:\n{result.stderr}"
    except subprocess.TimeoutExpired:
        return "Traceroute command timed out."

def main():
    print("Admin Network Diagnostic Utility")
    while True:
        host = input("Enter hostname or IP address (or 'exit' to quit): ").strip()
        if host.lower() == 'exit':
            print("Exiting.")
            break
        if not host:
            print("Please enter a valid hostname or IP.")
            continue
        print("Choose a diagnostic check:")
        print("1. Ping")
        print("2. Traceroute")
        choice = input("Enter 1 or 2: ").strip()
        if choice == '1':
            print(f"Running ping on {host}...\n")
            output = ping_host(host)
            print(output)
        elif choice == '2':
            print(f"Running traceroute on {host}...\n")
            output = traceroute_host(host)
            print(output)
        else:
            print("Invalid choice. Please enter 1 or 2.")
        print("\n---\n")

if __name__ == "__main__":
    main()