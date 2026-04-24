import subprocess

def ping_host(host):
    try:
        output = subprocess.check_output(['ping', '-c', '4', host], universal_newlines=True)
        return output
    except subprocess.CalledProcessError as e:
        return f"Ping failed:\n{e.output}"
    except Exception as e:
        return str(e)

def traceroute_host(host):
    try:
        output = subprocess.check_output(['traceroute', host], universal_newlines=True)
        return output
    except subprocess.CalledProcessError as e:
        return f"Traceroute failed:\n{e.output}"
    except Exception as e:
        return str(e)

def main():
    print("Server Dashboard Admin Utility")
    host = input("Enter hostname or IP address: ").strip()
    if not host:
        print("No host entered. Exiting.")
        return

    while True:
        print("\nChoose an action:")
        print("1. Ping")
        print("2. Traceroute")
        print("3. Exit")
        choice = input("Enter choice (1-3): ").strip()

        if choice == '1':
            print(f"\nPinging {host}...\n")
            print(ping_host(host))
        elif choice == '2':
            print(f"\nTraceroute to {host}...\n")
            print(traceroute_host(host))
        elif choice == '3':
            print("Exiting.")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()