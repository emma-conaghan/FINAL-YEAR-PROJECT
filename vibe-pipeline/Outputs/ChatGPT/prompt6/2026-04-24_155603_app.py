import subprocess

def run_ping(host):
    try:
        output = subprocess.check_output(['ping', '-c', '4', host], universal_newlines=True)
    except subprocess.CalledProcessError as e:
        output = e.output or "Ping failed."
    except Exception as e:
        output = str(e)
    return output

def run_traceroute(host):
    try:
        output = subprocess.check_output(['traceroute', host], universal_newlines=True)
    except subprocess.CalledProcessError as e:
        output = e.output or "Traceroute failed."
    except Exception as e:
        output = str(e)
    return output

def main():
    print("Server Dashboard Admin Utility")
    host = input("Enter hostname or IP address: ").strip()
    print("Run which check?")
    print("1. Ping")
    print("2. Traceroute")
    choice = input("Enter choice (1 or 2): ").strip()

    if choice == '1':
        print(f"\nPing results for {host}:\n")
        print(run_ping(host))
    elif choice == '2':
        print(f"\nTraceroute results for {host}:\n")
        print(run_traceroute(host))
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()