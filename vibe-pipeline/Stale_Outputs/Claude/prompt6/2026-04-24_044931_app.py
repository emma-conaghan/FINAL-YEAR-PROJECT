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
        output_widget.insert(tk.END, f"\n[Process finished with exit code {process.returncode}]")
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


def start_diagnostic(command_func, host, output_widget, status_label):
    host = host.strip()
    if not host:
        output_widget.config(state=tk.NORMAL)
        output_widget.delete(1.0, tk.END)
        output_widget.insert(tk.END, "[Error] Please enter a hostname or IP address.")
        output_widget.config(state=tk.DISABLED)
        return

    command = command_func(host)
    status_label.config(text=f"Status: Running {command[0]}...")

    def task():
        run_command(command, output_widget)
        status_label.config(text="Status: Idle")

    thread = threading.Thread(target=task, daemon=True)
    thread.start()


def build_gui():
    root = tk.Tk()
    root.title("Server Admin Dashboard - Network Diagnostics")
    root.geometry("800x600")
    root.resizable(True, True)

    style = ttk.Style()
    style.theme_use("clam")

    header = tk.Label(
        root,
        text="Network Diagnostic Tool",
        font=("Helvetica", 16, "bold"),
        bg="#2c3e50",
        fg="white",
        pady=10
    )
    header.pack(fill=tk.X)

    input_frame = tk.Frame(root, pady=8, padx=10)
    input_frame.pack(fill=tk.X)

    tk.Label(input_frame, text="Hostname / IP Address:", font=("Helvetica", 11)).pack(side=tk.LEFT)
    host_entry = tk.Entry(input_frame, width=40, font=("Helvetica", 11))
    host_entry.insert(0, "google.com")
    host_entry.pack(side=tk.LEFT, padx=8)

    button_frame = tk.Frame(root, pady=4, padx=10)
    button_frame.pack(fill=tk.X)

    status_label = tk.Label(root, text="Status: Idle", font=("Helvetica", 10), anchor="w", padx=10)
    status_label.pack(fill=tk.X)

    output_area = scrolledtext.ScrolledText(
        root,
        font=("Courier", 10),
        state=tk.DISABLED,
        bg="#1e1e1e",
        fg="#d4d4d4",
        insertbackground="white",
        padx=6,
        pady=6
    )
    output_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

    buttons = [
        ("Ping", get_ping_command),
        ("Traceroute", get_traceroute_command),
        ("NSLookup", get_nslookup_command),
        ("Whois", get_whois_command),
    ]

    for label, cmd_func in buttons:
        btn = tk.Button(
            button_frame,
            text=label,
            width=14,
            font=("Helvetica", 10, "bold"),
            bg="#2980b9",
            fg="white",
            activebackground="#1a6699",
            activeforeground="white",
            relief=tk.FLAT,
            padx=4,
            pady=4,
            command=lambda f=cmd_func: start_diagnostic(f, host_entry.get(), output_area, status_label)
        )
        btn.pack(side=tk.LEFT, padx=6)

    clear_btn = tk.Button(
        button_frame,
        text="Clear Output",
        width=14,
        font=("Helvetica", 10, "bold"),
        bg="#c0392b",
        fg="white",
        activebackground="#922b21",
        activeforeground="white",
        relief=tk.FLAT,
        padx=4,
        pady=4,
        command=lambda: (
            output_area.config(state=tk.NORMAL),
            output_area.delete(1.0, tk.END),
            output_area.config(state=tk.DISABLED)
        )
    )
    clear_btn.pack(side=tk.LEFT, padx=6)

    root.mainloop()


if __name__ == "__main__":
    build_gui()