import subprocess

def run_ping(host):
    try:
        output = subprocess.check_output(['ping', '-c', '4', host], stderr=subprocess.STDOUT, universal_newlines=True)
        return output
    except subprocess.CalledProcessError as e:
        return f"Ping failed:\n{e.output}"

def run_traceroute(host):
    try:
        output = subprocess.check_output(['traceroute', host], stderr=subprocess.STDOUT, universal_newlines=True)
        return output
    except subprocess.CalledProcessError as e:
        return f"Traceroute failed:\n{e.output}"

def main():
    print("Server Dashboard Admin Utility")
    while True:
        host = input("Enter hostname or IP address (or 'exit' to quit): ").strip()
        if host.lower() == 'exit':
            break
        if not host:
            print("Please enter a valid hostname or IP address.")
            continue

        print("Choose diagnostic check:")
        print("1. Ping")
        print("2. Traceroute")
        choice = input("Enter choice [1-2]: ").strip()

        if choice == '1':
            print("\nRunning ping...\n")
            result = run_ping(host)
        elif choice == '2':
            print("\nRunning traceroute...\n")
            result = run_traceroute(host)
        else:
            print("Invalid choice, please enter 1 or 2.")
            continue

        print(result)
        print("="*60)

if __name__ == "__main__":
    main()