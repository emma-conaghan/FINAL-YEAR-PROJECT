import subprocess

def run_ping(host):
    try:
        output = subprocess.check_output(
            ['ping', '-c', '4', host],
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        return output
    except subprocess.CalledProcessError as e:
        return f"Ping failed:\n{e.output}"
    except Exception as e:
        return f"Error: {str(e)}"

def run_traceroute(host):
    try:
        output = subprocess.check_output(
            ['traceroute', host],
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        return output
    except subprocess.CalledProcessError as e:
        return f"Traceroute failed:\n{e.output}"
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    print("Server Dashboard Admin Utility")
    host = input("Enter hostname or IP address: ").strip()
    if not host:
        print("No host provided. Exiting.")
        return

    while True:
        print("\nChoose a diagnostic check:")
        print("1. Ping")
        print("2. Traceroute")
        print("3. Exit")

        choice = input("Enter choice (1/2/3): ").strip()

        if choice == '1':
            print(f"\nRunning ping on {host}...\n")
            print(run_ping(host))
        elif choice == '2':
            print(f"\nRunning traceroute on {host}...\n")
            print(run_traceroute(host))
        elif choice == '3':
            print("Exiting.")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == '__main__':
    main()