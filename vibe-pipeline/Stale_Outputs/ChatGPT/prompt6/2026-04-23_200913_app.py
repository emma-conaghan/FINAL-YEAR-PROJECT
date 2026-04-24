import subprocess

def run_ping(host):
    try:
        # For Windows, use '-n', for Unix use '-c'
        count_param = '-n' if subprocess.os.name == 'nt' else '-c'
        result = subprocess.run(['ping', count_param, '4', host], capture_output=True, text=True, timeout=10)
        return result.stdout
    except Exception as e:
        return f"Error running ping: {e}"

def run_traceroute(host):
    try:
        if subprocess.os.name == 'nt':
            cmd = ['tracert', host]
        else:
            cmd = ['traceroute', host]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.stdout
    except Exception as e:
        return f"Error running traceroute: {e}"

def main():
    print("Server Admin Network Diagnostic Utility")
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
        choice = input("Enter 1 or 2: ").strip()
        if choice == '1':
            print(f"\nRunning ping on {host}...\n")
            output = run_ping(host)
            print(output)
        elif choice == '2':
            print(f"\nRunning traceroute on {host}...\n")
            output = run_traceroute(host)
            print(output)
        else:
            print("Invalid choice, please enter 1 or 2.")

if __name__ == '__main__':
    main()