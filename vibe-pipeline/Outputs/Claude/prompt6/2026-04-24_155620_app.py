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


def start_command_thread(command, output_widget, run_button):
    run_button.config(state=tk.DISABLED)

    def task():
        run_command(command, output_widget)
        run_button.config(state=tk.NORMAL)

    thread = threading.Thread(target=task, daemon=True)
    thread.start()


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


def on_run(action, host_entry, output_widget, run_button):
    host = host_entry.get().strip()
    if not host:
        output_widget.config(state=tk.NORMAL)
        output_widget.delete(1.0, tk.END)
        output_widget.insert(tk.END, "[Error] Please enter a hostname or IP address.")
        output_widget.config(state=tk.DISABLED)
        return

    if action == "ping":
        command = get_ping_command(host)
    elif action == "traceroute":
        command = get_traceroute_command(host)
    elif action == "nslookup":
        command = get_nslookup_command(host)
    elif action == "whois":
        command = get_whois_command(host)
    else:
        return

    start_command_thread(command, output_widget, run_button)


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
    host_entry.insert(0, "google.com")

    button_frame = tk.Frame(root, padx=10, pady=5)
    button_frame.pack(fill=tk.X)

    actions = [
        ("Ping", "ping"),
        ("Traceroute", "traceroute"),
        ("NSLookup", "nslookup"),
        ("Whois", "whois"),
    ]

    run_button_ref = [None]

    output_area = scrolledtext.ScrolledText(
        root,
        wrap=tk.WORD,
        font=("Courier", 10),
        bg="#1e1e1e",
        fg="#d4d4d4",
        insertbackground="white",
        state=tk.DISABLED,
        padx=5,
        pady=5
    )
    output_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

    def make_button(label, action):
        btn = tk.Button(
            button_frame,
            text=label,
            width=12,
            font=("Arial", 10),
            command=lambda: on_run(action, host_entry, output_area, run_button_ref[0])
        )
        btn.pack(side=tk.LEFT, padx=5)
        return btn

    first_button = None
    for label, action in actions:
        btn = make_button(label, action)
        if first_button is None:
            first_button = btn

    run_button_ref[0] = first_button

    clear_button = tk.Button(
        button_frame,
        text="Clear",
        width=10,
        font=("Arial", 10),
        command=lambda: (
            output_area.config(state=tk.NORMAL),
            output_area.delete(1.0, tk.END),
            output_area.config(state=tk.DISABLED)
        )
    )
    clear_button.pack(side=tk.RIGHT, padx=5)

    status_bar = tk.Label(
        root,
        text="Enter a hostname or IP and select a diagnostic tool.",
        bd=1,
        relief=tk.SUNKEN,
        anchor=tk.W,
        font=("Arial", 9)
    )
    status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    root.mainloop()


if __name__ == "__main__":
    build_gui()