import tkinter as tk
from tkinter import ttk, scrolledtext
import subprocess
import threading
import platform


def run_command(command, output_widget):
    output_widget.config(state=tk.NORMAL)
    output_widget.delete(1.0, tk.END)
    output_widget.insert(tk.END, f"Running: {' '.join(command)}\n\n")
    output_widget.update()

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        for line in process.stdout:
            output_widget.insert(tk.END, line)
            output_widget.see(tk.END)
            output_widget.update()
        process.wait()
        output_widget.insert(tk.END, f"\n[Process exited with code {process.returncode}]\n")
    except FileNotFoundError:
        output_widget.insert(tk.END, f"[Error] Command not found: {command[0]}\n")
    except Exception as e:
        output_widget.insert(tk.END, f"[Error] {str(e)}\n")

    output_widget.config(state=tk.DISABLED)


def run_in_thread(command, output_widget, run_button):
    run_button.config(state=tk.DISABLED)
    output_widget.config(state=tk.NORMAL)
    output_widget.delete(1.0, tk.END)
    output_widget.config(state=tk.DISABLED)

    def task():
        run_command(command, output_widget)
        run_button.config(state=tk.NORMAL)

    thread = threading.Thread(target=task, daemon=True)
    thread.start()


def get_ping_command(host, count=4):
    system = platform.system().lower()
    if system == "windows":
        return ["ping", "-n", str(count), host]
    else:
        return ["ping", "-c", str(count), host]


def get_traceroute_command(host):
    system = platform.system().lower()
    if system == "windows":
        return ["tracert", host]
    else:
        return ["traceroute", host]


def get_nslookup_command(host):
    return ["nslookup", host]


def get_whois_command(host):
    return ["whois", host]


def get_curl_headers_command(host):
    return ["curl", "-I", "--max-time", "10", host]


def build_gui():
    root = tk.Tk()
    root.title("Server Dashboard - Network Diagnostics")
    root.geometry("900x650")
    root.resizable(True, True)

    style = ttk.Style()
    style.theme_use("clam")

    header = tk.Label(
        root,
        text="Server Dashboard - Network Diagnostic Tool",
        font=("Helvetica", 16, "bold"),
        bg="#2c3e50",
        fg="white",
        pady=10
    )
    header.pack(fill=tk.X)

    input_frame = ttk.LabelFrame(root, text="Target Host", padding=10)
    input_frame.pack(fill=tk.X, padx=10, pady=5)

    tk.Label(input_frame, text="Hostname / IP Address:").grid(row=0, column=0, padx=5, sticky=tk.W)
    host_entry = ttk.Entry(input_frame, width=40, font=("Helvetica", 12))
    host_entry.grid(row=0, column=1, padx=5)
    host_entry.insert(0, "google.com")

    tk.Label(input_frame, text="Ping Count:").grid(row=0, column=2, padx=5, sticky=tk.W)
    ping_count_var = tk.StringVar(value="4")
    ping_count_entry = ttk.Entry(input_frame, textvariable=ping_count_var, width=5)
    ping_count_entry.grid(row=0, column=3, padx=5)

    buttons_frame = ttk.LabelFrame(root, text="Diagnostic Commands", padding=10)
    buttons_frame.pack(fill=tk.X, padx=10, pady=5)

    output_frame = ttk.LabelFrame(root, text="Output", padding=10)
    output_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    output_area = scrolledtext.ScrolledText(
        output_frame,
        wrap=tk.WORD,
        font=("Courier", 10),
        bg="#1e1e1e",
        fg="#00ff00",
        insertbackground="white",
        state=tk.DISABLED
    )
    output_area.pack(fill=tk.BOTH, expand=True)

    status_var = tk.StringVar(value="Ready")
    status_bar = tk.Label(root, textvariable=status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
    status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def get_host():
        return host_entry.get().strip()

    def get_count():
        try:
            return int(ping_count_var.get())
        except ValueError:
            return 4

    def do_ping():
        host = get_host()
        if not host:
            status_var.set("Please enter a hostname or IP address.")
            return
        status_var.set(f"Running ping on {host}...")
        cmd = get_ping_command(host, get_count())
        run_in_thread(cmd, output_area, ping_btn)

    def do_traceroute():
        host = get_host()
        if not host:
            status_var.set("Please enter a hostname or IP address.")
            return
        status_var.set(f"Running traceroute on {host}...")
        cmd = get_traceroute_command(host)
        run_in_thread(cmd, output_area, traceroute_btn)

    def do_nslookup():
        host = get_host()
        if not host:
            status_var.set("Please enter a hostname or IP address.")
            return
        status_var.set(f"Running nslookup on {host}...")
        cmd = get_nslookup_command(host)
        run_in_thread(cmd, output_area, nslookup_btn)

    def do_whois():
        host = get_host()
        if not host:
            status_var.set("Please enter a hostname or IP address.")
            return
        status_var.set(f"Running whois on {host}...")
        cmd = get_whois_command(host)
        run_in_thread(cmd, output_area, whois_btn)

    def do_curl_headers():
        host = get_host()
        if not host:
            status_var.set("Please enter a hostname or IP address.")
            return
        status_var.set(f"Fetching HTTP headers for {host}...")
        if not host.startswith("http://") and not host.startswith("https://"):
            host = "http://" + host
        cmd = get_curl_headers_command(host)
        run_in_thread(cmd, output_area, curl_btn)

    def clear_output():
        output_area.config(state=tk.NORMAL)
        output_area.delete(1.0, tk.END)
        output_area.config(state=tk.DISABLED)
        status_var.set("Output cleared.")

    ping_btn = ttk.Button(buttons_frame, text="Ping", command=do_ping, width=14)
    ping_btn.grid(row=0, column=0, padx=5, pady=5)

    traceroute_btn = ttk.Button(buttons_frame, text="Traceroute", command=do_traceroute, width=14)
    traceroute_btn.grid(row=0, column=1, padx=5, pady=5)

    nslookup_btn = ttk.Button(buttons_frame, text="NS Lookup", command=do_nslookup, width=14)
    nslookup_btn.grid(row=0, column=2, padx=5, pady=5)

    whois_btn = ttk.Button(buttons_frame, text="Whois", command=do_whois, width=14)
    whois_btn.grid(row=0, column=3, padx=5, pady=5)

    curl_btn = ttk.Button(buttons_frame, text="HTTP Headers", command=do_curl_headers, width=14)
    curl_btn.grid(row=0, column=4, padx=5, pady=5)

    clear_btn = ttk.Button(buttons_frame, text="Clear Output", command=clear_output, width=14)
    clear_btn.grid(row=0, column=5, padx=5, pady=5)

    root.mainloop()


if __name__ == "__main__":
    build_gui()