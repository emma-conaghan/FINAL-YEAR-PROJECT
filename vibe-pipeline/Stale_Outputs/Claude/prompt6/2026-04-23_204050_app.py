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
        output_widget.insert(tk.END, f"\nProcess finished with return code: {process.returncode}\n")
    except FileNotFoundError:
        output_widget.insert(tk.END, f"Error: Command not found - {command[0]}\n")
    except Exception as e:
        output_widget.insert(tk.END, f"Error: {str(e)}\n")

    output_widget.config(state=tk.DISABLED)


def run_in_thread(command, output_widget, run_button):
    run_button.config(state=tk.DISABLED)

    def task():
        run_command(command, output_widget)
        run_button.config(state=tk.NORMAL)

    thread = threading.Thread(target=task, daemon=True)
    thread.start()


def get_ping_command(host, count):
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

    top_frame = ttk.LabelFrame(root, text="Target Host", padding=10)
    top_frame.pack(fill=tk.X, padx=10, pady=10)

    ttk.Label(top_frame, text="Hostname / IP Address:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
    host_entry = ttk.Entry(top_frame, width=40)
    host_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
    host_entry.insert(0, "google.com")

    ttk.Label(top_frame, text="Ping Count:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
    ping_count_var = tk.StringVar(value="4")
    ping_count_entry = ttk.Entry(top_frame, textvariable=ping_count_var, width=5)
    ping_count_entry.grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)

    buttons_frame = ttk.LabelFrame(root, text="Diagnostic Tools", padding=10)
    buttons_frame.pack(fill=tk.X, padx=10, pady=5)

    output_frame = ttk.LabelFrame(root, text="Output", padding=10)
    output_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    output_area = scrolledtext.ScrolledText(
        output_frame,
        wrap=tk.WORD,
        font=("Courier", 10),
        bg="#1e1e1e",
        fg="#d4d4d4",
        insertbackground="white",
        state=tk.DISABLED
    )
    output_area.pack(fill=tk.BOTH, expand=True)

    def validate_host():
        host = host_entry.get().strip()
        if not host:
            output_area.config(state=tk.NORMAL)
            output_area.delete(1.0, tk.END)
            output_area.insert(tk.END, "Error: Please enter a hostname or IP address.\n")
            output_area.config(state=tk.DISABLED)
            return None
        return host

    def do_ping():
        host = validate_host()
        if host is None:
            return
        try:
            count = int(ping_count_var.get())
        except ValueError:
            count = 4
        command = get_ping_command(host, count)
        run_in_thread(command, output_area, ping_btn)

    def do_traceroute():
        host = validate_host()
        if host is None:
            return
        command = get_traceroute_command(host)
        run_in_thread(command, output_area, traceroute_btn)

    def do_nslookup():
        host = validate_host()
        if host is None:
            return
        command = get_nslookup_command(host)
        run_in_thread(command, output_area, nslookup_btn)

    def do_whois():
        host = validate_host()
        if host is None:
            return
        command = get_whois_command(host)
        run_in_thread(command, output_area, whois_btn)

    def do_curl_headers():
        host = validate_host()
        if host is None:
            return
        command = get_curl_headers_command(host)
        run_in_thread(command, output_area, curl_btn)

    def clear_output():
        output_area.config(state=tk.NORMAL)
        output_area.delete(1.0, tk.END)
        output_area.config(state=tk.DISABLED)

    ping_btn = ttk.Button(buttons_frame, text="Ping", command=do_ping, width=15)
    ping_btn.grid(row=0, column=0, padx=5, pady=5)

    traceroute_btn = ttk.Button(buttons_frame, text="Traceroute", command=do_traceroute, width=15)
    traceroute_btn.grid(row=0, column=1, padx=5, pady=5)

    nslookup_btn = ttk.Button(buttons_frame, text="NS Lookup", command=do_nslookup, width=15)
    nslookup_btn.grid(row=0, column=2, padx=5, pady=5)

    whois_btn = ttk.Button(buttons_frame, text="Whois", command=do_whois, width=15)
    whois_btn.grid(row=0, column=3, padx=5, pady=5)

    curl_btn = ttk.Button(buttons_frame, text="HTTP Headers", command=do_curl_headers, width=15)
    curl_btn.grid(row=0, column=4, padx=5, pady=5)

    clear_btn = ttk.Button(buttons_frame, text="Clear Output", command=clear_output, width=15)
    clear_btn.grid(row=0, column=5, padx=5, pady=5)

    status_bar = ttk.Label(root, text="Ready. Enter a hostname or IP and select a diagnostic tool.", relief=tk.SUNKEN, anchor=tk.W)
    status_bar.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=2)

    output_area.config(state=tk.NORMAL)
    output_area.insert(tk.END, "Welcome to the Server Dashboard - Network Diagnostics Tool\n")
    output_area.insert(tk.END, "=" * 60 + "\n")
    output_area.insert(tk.END, "Enter a hostname or IP address above and click a tool button.\n\n")
    output_area.insert(tk.END, f"Detected OS: {platform.system()} {platform.release()}\n")
    output_area.config(state=tk.DISABLED)

    root.mainloop()


if __name__ == "__main__":
    build_gui()