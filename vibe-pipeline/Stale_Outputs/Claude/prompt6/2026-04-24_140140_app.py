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
        output_widget.insert(tk.END, f"Error: Command not found. Make sure the tool is installed.\n")
    except Exception as e:
        output_widget.insert(tk.END, f"Error: {str(e)}\n")

    output_widget.config(state=tk.DISABLED)


def start_command_thread(command, output_widget, run_button):
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
    os_name = platform.system().lower()
    if os_name == "windows":
        return ["ping", "-n", str(count), host]
    else:
        return ["ping", "-c", str(count), host]


def get_traceroute_command(host):
    os_name = platform.system().lower()
    if os_name == "windows":
        return ["tracert", host]
    else:
        return ["traceroute", host]


def get_nslookup_command(host):
    return ["nslookup", host]


def get_whois_command(host):
    return ["whois", host]


def build_gui():
    root = tk.Tk()
    root.title("Server Dashboard - Network Diagnostics")
    root.geometry("800x600")
    root.resizable(True, True)

    top_frame = tk.Frame(root, padx=10, pady=10)
    top_frame.pack(fill=tk.X)

    tk.Label(top_frame, text="Hostname / IP Address:", font=("Arial", 11)).pack(side=tk.LEFT)

    host_entry = tk.Entry(top_frame, width=35, font=("Arial", 11))
    host_entry.pack(side=tk.LEFT, padx=(5, 10))
    host_entry.insert(0, "8.8.8.8")

    ping_count_label = tk.Label(top_frame, text="Ping Count:", font=("Arial", 11))
    ping_count_label.pack(side=tk.LEFT)

    ping_count_var = tk.StringVar(value="4")
    ping_count_entry = tk.Entry(top_frame, width=5, textvariable=ping_count_var, font=("Arial", 11))
    ping_count_entry.pack(side=tk.LEFT, padx=(5, 0))

    button_frame = tk.Frame(root, padx=10, pady=5)
    button_frame.pack(fill=tk.X)

    output_area = scrolledtext.ScrolledText(
        root,
        wrap=tk.WORD,
        font=("Courier", 10),
        state=tk.DISABLED,
        bg="#1e1e1e",
        fg="#d4d4d4",
        insertbackground="white",
        padx=5,
        pady=5
    )
    output_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))

    def on_ping():
        host = host_entry.get().strip()
        if not host:
            output_area.config(state=tk.NORMAL)
            output_area.delete(1.0, tk.END)
            output_area.insert(tk.END, "Please enter a hostname or IP address.\n")
            output_area.config(state=tk.DISABLED)
            return
        try:
            count = int(ping_count_var.get())
        except ValueError:
            count = 4
        command = get_ping_command(host, count)
        start_command_thread(command, output_area, ping_button)

    def on_traceroute():
        host = host_entry.get().strip()
        if not host:
            output_area.config(state=tk.NORMAL)
            output_area.delete(1.0, tk.END)
            output_area.insert(tk.END, "Please enter a hostname or IP address.\n")
            output_area.config(state=tk.DISABLED)
            return
        command = get_traceroute_command(host)
        start_command_thread(command, output_area, traceroute_button)

    def on_nslookup():
        host = host_entry.get().strip()
        if not host:
            output_area.config(state=tk.NORMAL)
            output_area.delete(1.0, tk.END)
            output_area.insert(tk.END, "Please enter a hostname or IP address.\n")
            output_area.config(state=tk.DISABLED)
            return
        command = get_nslookup_command(host)
        start_command_thread(command, output_area, nslookup_button)

    def on_whois():
        host = host_entry.get().strip()
        if not host:
            output_area.config(state=tk.NORMAL)
            output_area.delete(1.0, tk.END)
            output_area.insert(tk.END, "Please enter a hostname or IP address.\n")
            output_area.config(state=tk.DISABLED)
            return
        command = get_whois_command(host)
        start_command_thread(command, output_area, whois_button)

    def on_clear():
        output_area.config(state=tk.NORMAL)
        output_area.delete(1.0, tk.END)
        output_area.config(state=tk.DISABLED)

    button_style = {"font": ("Arial", 10), "width": 12, "pady": 4}

    ping_button = tk.Button(button_frame, text="Ping", command=on_ping, bg="#007acc", fg="white", **button_style)
    ping_button.pack(side=tk.LEFT, padx=5)

    traceroute_button = tk.Button(button_frame, text="Traceroute", command=on_traceroute, bg="#5c2d91", fg="white", **button_style)
    traceroute_button.pack(side=tk.LEFT, padx=5)

    nslookup_button = tk.Button(button_frame, text="NS Lookup", command=on_nslookup, bg="#107c10", fg="white", **button_style)
    nslookup_button.pack(side=tk.LEFT, padx=5)

    whois_button = tk.Button(button_frame, text="WHOIS", command=on_whois, bg="#d83b01", fg="white", **button_style)
    whois_button.pack(side=tk.LEFT, padx=5)

    clear_button = tk.Button(button_frame, text="Clear", command=on_clear, bg="#555555", fg="white", **button_style)
    clear_button.pack(side=tk.LEFT, padx=5)

    status_bar = tk.Label(root, text="Ready. Enter a hostname or IP and select a diagnostic tool.", anchor=tk.W, font=("Arial", 9), bd=1, relief=tk.SUNKEN)
    status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    root.mainloop()


if __name__ == "__main__":
    build_gui()