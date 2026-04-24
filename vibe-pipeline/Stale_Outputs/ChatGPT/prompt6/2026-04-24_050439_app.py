import subprocess

def ping(host):
    try:
        result = subprocess.run(['ping', '-c', '4', host], capture_output=True, text=True)
        return result.stdout if result.returncode == 0 else f"Ping failed:\n{result.stderr}"
    except Exception as e:
        return f"Error running ping: {e}"

def traceroute(host):
    try:
        result = subprocess.run(['traceroute', host], capture_output=True, text=True)
        return result.stdout if result.returncode == 0 else f"Traceroute failed:\n{result.stderr}"
    except Exception as e:
        return f"Error running traceroute: {e}"

def main():
    print("Server Dashboard Network Diagnostic Utility")
    while True:
        host = input("\nEnter hostname or IP address (or 'exit' to quit): ").strip()
        if host.lower() == 'exit':
            print("Exiting.")
            break
        if not host:
            print("Please enter a valid hostname or IP address.")
            continue
        print("\nChoose diagnostic check:")
        print("1. Ping")
        print("2. Traceroute")
        choice = input("Enter choice (1 or 2): ").strip()
        if choice == '1':
            print(f"\nPinging {host}...")
            output = ping(host)
            print(output)
        elif choice == '2':
            print(f"\nTracerouting {host}...")
            output = traceroute(host)
            print(output)
        else:
            print("Invalid choice. Please enter 1 or 2.")

if __name__ == '__main__':
    main()