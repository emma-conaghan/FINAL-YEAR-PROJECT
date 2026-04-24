import subprocess

def run_ping(host):
    try:
        result = subprocess.run(['ping', '-c', '4', host], capture_output=True, text=True, timeout=10)
        return result.stdout if result.returncode == 0 else f"Ping failed:\n{result.stderr}"
    except Exception as e:
        return f"Error running ping: {e}"

def run_traceroute(host):
    try:
        result = subprocess.run(['traceroute', host], capture_output=True, text=True, timeout=30)
        return result.stdout if result.returncode == 0 else f"Traceroute failed:\n{result.stderr}"
    except Exception as e:
        return f"Error running traceroute: {e}"

def main():
    print("Server Admin Utility - Network Diagnostic Checks")
    while True:
        host = input("\nEnter hostname or IP address (or 'quit' to exit): ").strip()
        if host.lower() == 'quit':
            break
        if not host:
            print("Please enter a valid hostname or IP address.")
            continue
        
        print("\nSelect diagnostic check:")
        print("1. Ping")
        print("2. Traceroute")
        choice = input("Choice (1 or 2): ").strip()
        
        if choice == '1':
            output = run_ping(host)
            print(f"\nPing results for {host}:\n{output}")
        elif choice == '2':
            output = run_traceroute(host)
            print(f"\nTraceroute results for {host}:\n{output}")
        else:
            print("Invalid choice. Please select 1 or 2.")

if __name__ == '__main__':
    main()