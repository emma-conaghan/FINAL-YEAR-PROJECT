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


def start_command_thread(command, output_widget):
    thread = threading.Thread(target=run_command, args=(command, output_widget), daemon=True)
    thread.start()


def build_ping_command(host, count):
    os_name = platform.system().lower()
    if os_name == "windows":
        return ["ping", "-n", str(count), host]
    else:
        return ["ping", "-c", str(count), host]


def build_traceroute_command(host):
    os_name = platform.system().lower()
    if os_name == "windows":
        return ["tracert", host]
    else:
        return ["traceroute", host]


def build_nslookup_command(host):
    return ["nslookup", host]


def build_whois_command(host):
    return ["whois", host]


def run_ping(host_entry, count_entry, output_widget):
    host = host_entry.get().strip()
    if not host:
        show_error("Please enter a hostname or IP address.", output_widget)
        return
    try:
        count = int(count_entry.get().strip())
        if count < 1:
            raise ValueError
    except ValueError:
        show_error("Please enter a valid ping count (positive integer).", output_widget)
        return
    command = build_ping_command(host, count)
    start_command_thread(command, output_widget)


def run_traceroute(host_entry, output_widget):
    host = host_entry.get().strip()
    if not host:
        show_error("Please enter a hostname or IP address.", output_widget)
        return
    command = build_traceroute_command(host)
    start_command_thread(command, output_widget)


def run_nslookup(host_entry, output_widget):
    host = host_entry.get().strip()
    if not host:
        show_error("Please enter a hostname or IP address.", output_widget)
        return
    command = build_nslookup_command(host)
    start_command_thread(command, output_widget)


def run_whois(host_entry, output_widget):
    host = host_entry.get().strip()
    if not host:
        show_error("Please enter a hostname or IP address.", output_widget)
        return
    command = build_whois_command(host)
    start_command_thread(command, output_widget)


def show_error(message, output_widget):
    output_widget.config(state=tk.NORMAL)
    output_widget.delete(1.0, tk.END)
    output_widget.insert(tk.END, f"[Error] {message}")
    output_widget.config(state=tk.DISABLED)


def clear_output(output_widget):
    output_widget.config(state=tk.NORMAL)
    output_widget.delete(1.0, tk.END)
    output_widget.config(state=tk.DISABLED)


def main():
    root = tk.Tk()
    root.title("Server Dashboard - Network Diagnostics")
    root.geometry("800x600")
    root.resizable(True, True)

    style = ttk.Style()
    style.theme_use("clam")

    header_frame = tk.Frame(root, bg="#2c3e50", pady=10)
    header_frame.pack(fill=tk.X)

    header_label = tk.Label(
        header_frame,
        text="Server Dashboard - Network Diagnostics",
        font=("Helvetica", 16, "bold"),
        bg="#2c3e50",
        fg="white"
    )
    header_label.pack()

    input_frame = tk.LabelFrame(root, text="Target", padx=10, pady=10, font=("Helvetica", 10, "bold"))
    input_frame.pack(fill=tk.X, padx=10, pady=10)

    tk.Label(input_frame, text="Hostname / IP Address:", font=("Helvetica", 10)).grid(row=0, column=0, sticky=tk.W)
    host_entry = tk.Entry(input_frame, width=40, font=("Helvetica", 10))
    host_entry.grid(row=0, column=1, padx=10, pady=5)
    host_entry.insert(0, "google.com")

    tk.Label(input_frame, text="Ping Count:", font=("Helvetica", 10)).grid(row=0, column=2, sticky=tk.W)
    count_entry = tk.Entry(input_frame, width=5, font=("Helvetica", 10))
    count_entry.grid(row=0, column=3, padx=5)
    count_entry.insert(0, "4")

    buttons_frame = tk.LabelFrame(root, text="Diagnostics", padx=10, pady=10, font=("Helvetica", 10, "bold"))
    buttons_frame.pack(fill=tk.X, padx=10)

    output_widget = scrolledtext.ScrolledText(
        root,
        font=("Courier", 10),
        state=tk.DISABLED,
        bg="#1e1e1e",
        fg="#d4d4d4",
        insertbackground="white"
    )
    output_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    button_configs = [
        ("Ping", lambda: run_ping(host_entry, count_entry, output_widget), "#27ae60"),
        ("Traceroute", lambda: run_traceroute(host_entry, output_widget), "#2980b9"),
        ("NSLookup", lambda: run_nslookup(host_entry, output_widget), "#8e44ad"),
        ("Whois", lambda: run_whois(host_entry, output_widget), "#e67e22"),
        ("Clear", lambda: clear_output(output_widget), "#7f8c8d"),
    ]

    for idx, (label, command, color) in enumerate(button_configs):
        btn = tk.Button(
            buttons_frame,
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
        btn.grid(row=0, column=idx, padx=5, pady=5)

    status_bar = tk.Label(
        root,
        text="Ready. Enter a hostname or IP address and select a diagnostic tool.",
        bd=1,
        relief=tk.SUNKEN,
        anchor=tk.W,
        font=("Helvetica", 9)
    )
    status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    root.mainloop()


if __name__ == "__main__":
    main()