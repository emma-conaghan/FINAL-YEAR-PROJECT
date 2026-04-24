import subprocess

def run_ping(host):
    try:
        output = subprocess.check_output(
            ['ping', '-c', '4', host], 
            stderr=subprocess.STDOUT, 
            universal_newlines=True
        )
    except subprocess.CalledProcessError as e:
        output = e.output
    return output

def run_traceroute(host):
    try:
        output = subprocess.check_output(
            ['traceroute', host], 
            stderr=subprocess.STDOUT, 
            universal_newlines=True
        )
    except subprocess.CalledProcessError as e:
        output = e.output
    return output

def main():
    while True:
        print("Network Diagnostic Utility")
        host = input("Enter hostname or IP address (or 'exit' to quit): ").strip()
        if host.lower() == 'exit':
            break

        print("Choose diagnostic check:")
        print("1. Ping")
        print("2. Traceroute")
        choice = input("Enter choice (1 or 2): ").strip()

        if choice == '1':
            print(f"Pinging {host}...\n")
            result = run_ping(host)
            print(result)
        elif choice == '2':
            print(f"Tracerouting {host}...\n")
            result = run_traceroute(host)
            print(result)
        else:
            print("Invalid choice. Please select 1 or 2.")

        print("-" * 40)

if __name__ == "__main__":
    main()