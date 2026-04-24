import subprocess

def run_ping(host):
    try:
        output = subprocess.check_output(["ping", "-c", "4", host], stderr=subprocess.STDOUT, universal_newlines=True)
    except subprocess.CalledProcessError as e:
        output = e.output
    return output

def run_traceroute(host):
    try:
        output = subprocess.check_output(["traceroute", host], stderr=subprocess.STDOUT, universal_newlines=True)
    except subprocess.CalledProcessError as e:
        output = e.output
    return output

def main():
    print("Network Diagnostic Utility")
    host = input("Enter a hostname or IP address: ").strip()
    if not host:
        print("No hostname or IP entered. Exiting.")
        return
    while True:
        print("\nChoose a diagnostic check:")
        print("1. Ping")
        print("2. Traceroute")
        print("3. Exit")
        choice = input("Enter choice (1-3): ").strip()
        if choice == "1":
            print(f"\nPinging {host}...\n")
            print(run_ping(host))
        elif choice == "2":
            print(f"\nTracerouting {host}...\n")
            print(run_traceroute(host))
        elif choice == "3":
            print("Exiting.")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()