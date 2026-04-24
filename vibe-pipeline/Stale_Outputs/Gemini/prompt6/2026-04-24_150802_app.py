import subprocess
import platform
import sys

def execute_network_command(cmd_string):
    try:
        # Use shell=True to allow command execution across different OS environments
        # capture_output/text are Python 3.7+ features
        process = subprocess.run(
            cmd_string, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=60
        )
        return process.stdout if process.returncode == 0 else process.stderr
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 60 seconds."
    except Exception as e:
        return f"Unexpected error: {str(e)}"

def ping_target(host):
    # Windows uses -n, Unix-like systems use -c
    param = "-n 4" if platform.system().lower() == "windows" else "-c 4"
    command = f"ping {param} {host}"
    return execute_network_command(command)

def trace_target(host):
    # Windows uses tracert, Unix-like systems use traceroute
    command = "tracert" if platform.system().lower() == "windows" else "traceroute"
    return execute_network_command(f"{command} {host}")

def display_menu():
    print("\n" + "="*40)
    print("SERVER ADMIN NETWORK DASHBOARD")
    print("="*40)
    print("1. Ping Host")
    print("2. Traceroute")
    print("3. Exit")
    print("-"*40)

def main():
    while True:
        display_menu()
        choice = input("Select an option (1-3): ").strip()
        
        if choice == '3':
            print("Exiting utility...")
            sys.exit(0)
            
        if choice not in ['1', '2']:
            print("Invalid selection. Please enter 1, 2, or 3.")
            continue
            
        target = input("Enter Hostname or IP Address: ").strip()
        
        if not target:
            print("Error: Target cannot be empty.")
            continue

        if choice == '1':
            print(f"\n[Action] Pinging {target}...")
            results = ping_target(target)
            print(results)
        elif choice == '2':
            print(f"\n[Action] Running traceroute to {target}...")
            results = trace_target(target)
            print(results)

if __name__ == "__main__":
    main()