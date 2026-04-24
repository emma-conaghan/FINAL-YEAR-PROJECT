import subprocess

def run_ping(host):
    try:
        result = subprocess.run(['ping', '-c', '4', host], capture_output=True, text=True, timeout=10)
        return result.stdout if result.returncode == 0 else f"Ping failed:\n{result.stderr}"
    except Exception as e:
        return f"Error running ping: {e}"

def run_traceroute(host):
    try:
        # Use traceroute or tracert based on platform
        import platform
        cmd = ['traceroute', host] if platform.system() != 'Windows' else ['tracert', host]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.stdout if result.returncode == 0 else f"Traceroute failed:\n{result.stderr}"
    except Exception as e:
        return f"Error running traceroute: {e}"

def main():
    while True:
        host = input("Enter hostname or IP address (or 'exit' to quit): ").strip()
        if host.lower() == 'exit':
            break
        if not host:
            print("Please enter a valid hostname or IP.")
            continue
        
        print("\nChoose a diagnostic:\n1) Ping\n2) Traceroute\n3) Both")
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == '1':
            print(f"\nRunning ping on {host}...\n")
            print(run_ping(host))
        elif choice == '2':
            print(f"\nRunning traceroute on {host}...\n")
            print(run_traceroute(host))
        elif choice == '3':
            print(f"\nRunning ping on {host}...\n")
            print(run_ping(host))
            print(f"\nRunning traceroute on {host}...\n")
            print(run_traceroute(host))
        else:
            print("Invalid choice, please select 1, 2, or 3.")

        print("\n" + "="*60 + "\n")

if __name__ == '__main__':
    main()