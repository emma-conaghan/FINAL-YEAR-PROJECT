import subprocess

def run_ping(host):
    try:
        output = subprocess.check_output(["ping", "-c", "4", host], stderr=subprocess.STDOUT, text=True)
        return output
    except subprocess.CalledProcessError as e:
        return f"Ping failed:\n{e.output}"

def run_traceroute(host):
    try:
        output = subprocess.check_output(["traceroute", host], stderr=subprocess.STDOUT, text=True)
        return output
    except subprocess.CalledProcessError as e:
        return f"Traceroute failed:\n{e.output}"

def main():
    print("Admin Network Diagnostic Utility")
    while True:
        host = input("Enter hostname or IP (or 'exit' to quit): ").strip()
        if host.lower() == "exit":
            break

        print("Choose an option:")
        print("1. Ping")
        print("2. Traceroute")

        choice = input("Enter option number: ").strip()
        if choice == "1":
            result = run_ping(host)
            print(f"Ping results for {host}:\n{result}")
        elif choice == "2":
            result = run_traceroute(host)
            print(f"Traceroute results for {host}:\n{result}")
        else:
            print("Invalid option selected.")

        print("\n" + "="*40 + "\n")

if __name__ == "__main__":
    main()