import subprocess

def ping(host):
    try:
        output = subprocess.check_output(['ping', '-c', '4', host], text=True)
        return output
    except subprocess.CalledProcessError as e:
        return f"Ping failed:\n{e.output}"

def traceroute(host):
    try:
        output = subprocess.check_output(['traceroute', host], text=True)
        return output
    except subprocess.CalledProcessError as e:
        return f"Traceroute failed:\n{e.output}"
    except FileNotFoundError:
        return "Traceroute command not found on this system."

def main():
    print("Server Dashboard Admin Utility")
    while True:
        host = input("Enter hostname or IP address (or 'exit' to quit): ").strip()
        if host.lower() == "exit":
            break
        if not host:
            print("Please enter a valid hostname or IP address")
            continue

        print("Choose diagnostic check:")
        print("1) Ping")
        print("2) Traceroute")
        choice = input("Enter choice number: ").strip()

        if choice == "1":
            print(f"Running ping on {host}...\n")
            result = ping(host)
            print(result)
        elif choice == "2":
            print(f"Running traceroute on {host}...\n")
            result = traceroute(host)
            print(result)
        else:
            print("Invalid choice, please select 1 or 2")
        print("-" * 40)

if __name__ == "__main__":
    main()