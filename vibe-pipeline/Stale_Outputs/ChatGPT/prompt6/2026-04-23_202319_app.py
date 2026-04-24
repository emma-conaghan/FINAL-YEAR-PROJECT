import subprocess

def run_command(command):
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True, timeout=10)
        if result.returncode == 0:
            return result.stdout
        else:
            return f"Error:\n{result.stderr}"
    except Exception as e:
        return f"Exception occurred: {str(e)}"

def ping_host(host):
    # Ping 4 times, works on Windows and Unix
    import platform
    param = "-n" if platform.system().lower() == "windows" else "-c"
    command = f"ping {param} 4 {host}"
    return run_command(command)

def traceroute_host(host):
    import platform
    system = platform.system().lower()
    if system == "windows":
        command = f"tracert {host}"
    else:
        command = f"traceroute {host}"
    return run_command(command)

def main():
    print("Server Dashboard Admin Utility")
    while True:
        host = input("Enter a hostname or IP address (or 'exit' to quit): ").strip()
        if host.lower() == "exit":
            break
        if not host:
            print("Please enter a valid hostname or IP address.")
            continue
        
        print("\nChoose a network diagnostic check:")
        print("1 - Ping")
        print("2 - Traceroute")
        choice = input("Enter choice number: ").strip()
        if choice == "1":
            output = ping_host(host)
        elif choice == "2":
            output = traceroute_host(host)
        else:
            print("Invalid choice. Please try again.")
            continue

        print(f"\nOutput for {host}:\n{'-'*40}")
        print(output)
        print('-'*40 + '\n')

if __name__ == "__main__":
    main()