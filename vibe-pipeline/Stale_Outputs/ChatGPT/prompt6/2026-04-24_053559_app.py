import subprocess

def run_ping(host):
    try:
        output = subprocess.check_output(['ping', '-c', '4', host], stderr=subprocess.STDOUT, universal_newlines=True)
    except subprocess.CalledProcessError as e:
        output = e.output
    return output

def run_traceroute(host):
    try:
        output = subprocess.check_output(['traceroute', host], stderr=subprocess.STDOUT, universal_newlines=True)
    except subprocess.CalledProcessError as e:
        output = e.output
    return output

def main():
    print("Simple Server Admin Utility")
    while True:
        host = input("Enter hostname or IP address (or 'exit' to quit): ").strip()
        if host.lower() == 'exit':
            break
        if not host:
            print("Please enter a valid hostname or IP address.")
            continue
        print("Choose a diagnostic check:")
        print("1. Ping")
        print("2. Traceroute")
        choice = input("Enter your choice (1 or 2): ").strip()
        if choice == '1':
            print(f"\n--- Pinging {host} ---")
            print(run_ping(host))
        elif choice == '2':
            print(f"\n--- Traceroute to {host} ---")
            print(run_traceroute(host))
        else:
            print("Invalid choice. Please select 1 or 2.")
        print("\n")

if __name__ == '__main__':
    main()