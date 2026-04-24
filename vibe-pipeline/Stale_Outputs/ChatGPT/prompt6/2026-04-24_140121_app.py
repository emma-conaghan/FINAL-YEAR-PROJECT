import subprocess

def run_ping(host):
    try:
        output = subprocess.check_output(["ping", "-c", "4", host], universal_newlines=True)
    except subprocess.CalledProcessError as e:
        output = e.output or "Ping failed."
    except Exception as e:
        output = str(e)
    return output

def run_traceroute(host):
    try:
        output = subprocess.check_output(["traceroute", host], universal_newlines=True)
    except subprocess.CalledProcessError as e:
        output = e.output or "Traceroute failed."
    except Exception as e:
        output = str(e)
    return output

def main():
    while True:
        print("Server Admin Utility")
        print("1. Ping a host")
        print("2. Traceroute a host")
        print("3. Exit")
        choice = input("Enter your choice: ").strip()

        if choice == "1":
            host = input("Enter hostname or IP address to ping: ").strip()
            print("\nRunning ping...\n")
            print(run_ping(host))
            print("\n")
        elif choice == "2":
            host = input("Enter hostname or IP address for traceroute: ").strip()
            print("\nRunning traceroute...\n")
            print(run_traceroute(host))
            print("\n")
        elif choice == "3":
            print("Exiting.")
            break
        else:
            print("Invalid choice, please try again.\n")

if __name__ == "__main__":
    main()