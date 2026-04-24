import subprocess

def ping(host):
    try:
        output = subprocess.check_output(['ping', '-c', '4', host], stderr=subprocess.STDOUT, universal_newlines=True)
    except subprocess.CalledProcessError as e:
        output = e.output
    return output

def traceroute(host):
    try:
        output = subprocess.check_output(['traceroute', host], stderr=subprocess.STDOUT, universal_newlines=True)
    except subprocess.CalledProcessError as e:
        output = e.output
    return output

def main():
    print("Server Dashboard Admin Utility")
    print("==============================")
    host = input("Enter hostname or IP address: ").strip()
    while True:
        print("\nChoose a check to run:")
        print("1. Ping")
        print("2. Traceroute")
        print("3. Exit")
        choice = input("Your choice: ").strip()
        if choice == '1':
            print(f"\n--- Pinging {host} ---")
            print(ping(host))
        elif choice == '2':
            print(f"\n--- Traceroute to {host} ---")
            print(traceroute(host))
        elif choice == '3':
            print("Exiting.")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == '__main__':
    main()