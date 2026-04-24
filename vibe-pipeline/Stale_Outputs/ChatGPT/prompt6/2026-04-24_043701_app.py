import subprocess

def run_ping(target):
    try:
        output = subprocess.check_output(['ping', '-c', '4', target], stderr=subprocess.STDOUT, text=True)
    except subprocess.CalledProcessError as e:
        output = e.output
    return output

def run_traceroute(target):
    try:
        output = subprocess.check_output(['traceroute', target], stderr=subprocess.STDOUT, text=True)
    except subprocess.CalledProcessError as e:
        output = e.output
    return output

def main():
    print("=== Network Diagnostic Tool ===")
    while True:
        target = input("Enter hostname or IP address (or 'exit' to quit): ").strip()
        if target.lower() == 'exit':
            break
        print("Choose a diagnostic to run:")
        print("1. Ping")
        print("2. Traceroute")
        choice = input("Enter 1 or 2: ").strip()
        if choice == '1':
            print(f"\nPinging {target}...\n")
            result = run_ping(target)
            print(result)
        elif choice == '2':
            print(f"\nTracerouting {target}...\n")
            result = run_traceroute(target)
            print(result)
        else:
            print("Invalid choice. Please enter 1 or 2.")
        print("-" * 40)

if __name__ == '__main__':
    main()