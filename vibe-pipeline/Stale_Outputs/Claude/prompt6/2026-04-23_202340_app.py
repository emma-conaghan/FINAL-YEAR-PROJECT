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
        output_widget.insert(tk.END, f"\n[Process exited with code {process.returncode}]")
    except FileNotFoundError:
        output_widget.insert(tk.END, f"[Error] Command not found: {command[0]}")
    except Exception as e:
        output_widget.insert(tk.END, f"[Error] {str(e)}")

    output_widget.config(state=tk.DISABLED)


def get_ping_command(host):
    system = platform.system().lower()
    if system == "windows":
        return ["ping", "-n", "4", host]
    else:
        return ["ping", "-c", "4", host]


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


def start_diagnostic(command, output_widget):
    thread = threading.Thread(target=run_command, args=(command, output_widget), daemon=True)
    thread.start()


def build_gui():
    root = tk.Tk()
    root.title("Server Dashboard - Network Diagnostics")
    root.geometry("800x600")
    root.resizable(True, True)

    header = tk.Label(root, text="Network Diagnostic Utility", font=("Helvetica", 16, "bold"), bg="#2c3e50", fg="white")
    header.pack(fill=tk.X, pady=0)

    input_frame = tk.Frame(root, pady=10, padx=10)
    input_frame.pack(fill=tk.X)

    tk.Label(input_frame, text="Hostname / IP Address:", font=("Helvetica", 11)).grid(row=0, column=0, sticky=tk.W, padx=5)
    host_entry = tk.Entry(input_frame, width=40, font=("Helvetica", 11))
    host_entry.grid(row=0, column=1, padx=5)
    host_entry.insert(0, "google.com")

    button_frame = tk.Frame(root, pady=5, padx=10)
    button_frame.pack(fill=tk.X)

    output_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Courier", 10), state=tk.DISABLED, bg="#1e1e1e", fg="#d4d4d4", insertbackground="white")
    output_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def on_ping():
        host = host_entry.get().strip()
        if host:
            cmd = get_ping_command(host)
            start_diagnostic(cmd, output_area)

    def on_traceroute():
        host = host_entry.get().strip()
        if host:
            cmd = get_traceroute_command(host)
            start_diagnostic(cmd, output_area)

    def on_nslookup():
        host = host_entry.get().strip()
        if host:
            cmd = get_nslookup_command(host)
            start_diagnostic(cmd, output_area)

    def on_whois():
        host = host_entry.get().strip()
        if host:
            cmd = get_whois_command(host)
            start_diagnostic(cmd, output_area)

    def on_clear():
        output_area.config(state=tk.NORMAL)
        output_area.delete(1.0, tk.END)
        output_area.config(state=tk.DISABLED)

    buttons = [
        ("Ping", on_ping, "#27ae60"),
        ("Traceroute", on_traceroute, "#2980b9"),
        ("NSLookup", on_nslookup, "#8e44ad"),
        ("Whois", on_whois, "#e67e22"),
        ("Clear Output", on_clear, "#c0392b"),
    ]

    for label, command, color in buttons:
        btn = tk.Button(
            button_frame,
            text=label,
            command=command,
            bg=color,
            fg="white",
            font=("Helvetica", 10, "bold"),
            padx=10,
            pady=5,
            relief=tk.FLAT,
            cursor="hand2"
        )
        btn.pack(side=tk.LEFT, padx=5)

    status_bar = tk.Label(root, text="Ready. Enter a hostname or IP and select a diagnostic.", bd=1, relief=tk.SUNKEN, anchor=tk.W, font=("Helvetica", 9))
    status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    root.mainloop()


if __name__ == "__main__":
    build_gui()