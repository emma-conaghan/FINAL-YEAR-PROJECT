import subprocess

def run_ping(host):
    try:
        result = subprocess.run(
            ["ping", "-c", "4", host],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            return result.stdout
        else:
            return f"Ping failed:\n{result.stderr}"
    except Exception as e:
        return f"Error running ping: {e}"

def run_traceroute(host):
    try:
        result = subprocess.run(
            ["traceroute", host],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            return result.stdout
        else:
            return f"Traceroute failed:\n{result.stderr}"
    except Exception as e:
        return f"Error running traceroute: {e}"

def main():
    print("Server Dashboard Admin Utility")
    host = input("Enter hostname or IP address to check: ").strip()
    if not host:
        print("No host entered. Exiting.")
        return

    print("\nRunning ping...")
    ping_output = run_ping(host)
    print(ping_output)

    print("\nRunning traceroute...")
    trac_output = run_traceroute(host)
    print(trac_output)

if __name__ == "__main__":
    main()