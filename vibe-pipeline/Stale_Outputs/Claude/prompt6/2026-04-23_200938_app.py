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
    except FileNotFoundError as e:
        output_widget.insert(tk.END, f"Error: Command not found. {e}\n")
    except Exception as e:
        output_widget.insert(tk.END, f"Unexpected error: {e}\n")

    output_widget.config(state=tk.DISABLED)


def get_ping_command(host, count):
    system = platform.system()
    if system == "Windows":
        return ["ping", "-n", str(count), host]
    else:
        return ["ping", "-c", str(count), host]


def get_traceroute_command(host):
    system = platform.system()
    if system == "Windows":
        return ["tracert", host]
    else:
        return ["traceroute", host]


def get_nslookup_command(host):
    return ["nslookup", host]


def get_netstat_command():
    system = platform.system()
    if system == "Windows":
        return ["netstat", "-an"]
    else:
        return ["netstat", "-an"]


def start_command(command, output_widget, run_button):
    run_button.config(state=tk.DISABLED)
    output_widget.config(state=tk.NORMAL)
    output_widget.delete(1.0, tk.END)
    output_widget.insert(tk.END, "Starting...\n")
    output_widget.config(state=tk.DISABLED)

    def task():
        run_command(command, output_widget)
        run_button.config(state=tk.NORMAL)

    thread = threading.Thread(target=task, daemon=True)
    thread.start()


def build_gui():
    root = tk.Tk()
    root.title("Server Dashboard - Network Diagnostics")
    root.geometry("900x700")
    root.resizable(True, True)

    style = ttk.Style()
    style.theme_use("clam")

    title_label = tk.Label(
        root,
        text="Server Dashboard - Network Diagnostic Tool",
        font=("Helvetica", 16, "bold"),
        bg="#2c3e50",
        fg="white",
        pady=10
    )
    title_label.pack(fill=tk.X)

    input_frame = tk.Frame(root, bg="#ecf0f1", pady=10, padx=10)
    input_frame.pack(fill=tk.X)

    tk.Label(input_frame, text="Hostname / IP Address:", bg="#ecf0f1", font=("Helvetica", 11)).grid(
        row=0, column=0, sticky=tk.W, padx=5
    )
    host_entry = tk.Entry(input_frame, width=35, font=("Helvetica", 11))
    host_entry.insert(0, "8.8.8.8")
    host_entry.grid(row=0, column=1, padx=5)

    tk.Label(input_frame, text="Ping Count:", bg="#ecf0f1", font=("Helvetica", 11)).grid(
        row=0, column=2, sticky=tk.W, padx=5
    )
    ping_count_var = tk.StringVar(value="4")
    ping_count_entry = tk.Entry(input_frame, textvariable=ping_count_var, width=5, font=("Helvetica", 11))
    ping_count_entry.grid(row=0, column=3, padx=5)

    button_frame = tk.Frame(root, bg="#ecf0f1", pady=5, padx=10)
    button_frame.pack(fill=tk.X)

    output_area = scrolledtext.ScrolledText(
        root,
        wrap=tk.WORD,
        font=("Courier", 10),
        bg="#1e1e1e",
        fg="#dcdcdc",
        insertbackground="white",
        state=tk.DISABLED,
        padx=10,
        pady=10
    )
    output_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def on_ping():
        host = host_entry.get().strip()
        if not host:
            show_error("Please enter a hostname or IP address.")
            return
        try:
            count = int(ping_count_var.get())
            if count < 1:
                raise ValueError
        except ValueError:
            show_error("Ping count must be a positive integer.")
            return
        command = get_ping_command(host, count)
        start_command(command, output_area, ping_button)

    def on_traceroute():
        host = host_entry.get().strip()
        if not host:
            show_error("Please enter a hostname or IP address.")
            return
        command = get_traceroute_command(host)
        start_command(command, output_area, traceroute_button)

    def on_nslookup():
        host = host_entry.get().strip()
        if not host:
            show_error("Please enter a hostname or IP address.")
            return
        command = get_nslookup_command(host)
        start_command(command, output_area, nslookup_button)

    def on_netstat():
        command = get_netstat_command()
        start_command(command, output_area, netstat_button)

    def on_clear():
        output_area.config(state=tk.NORMAL)
        output_area.delete(1.0, tk.END)
        output_area.config(state=tk.DISABLED)

    def show_error(message):
        output_area.config(state=tk.NORMAL)
        output_area.delete(1.0, tk.END)
        output_area.insert(tk.END, f"Error: {message}\n")
        output_area.config(state=tk.DISABLED)

    button_style = {
        "font": ("Helvetica", 11),
        "padx": 10,
        "pady": 5,
        "cursor": "hand2",
        "relief": tk.FLAT,
        "bd": 0
    }

    ping_button = tk.Button(
        button_frame,
        text="Ping",
        bg="#27ae60",
        fg="white",
        command=on_ping,
        **button_style
    )
    ping_button.grid(row=0, column=0, padx=5)

    traceroute_button = tk.Button(
        button_frame,
        text="Traceroute",
        bg="#2980b9",
        fg="white",
        command=on_traceroute,
        **button_style
    )
    traceroute_button.grid(row=0, column=1, padx=5)

    nslookup_button = tk.Button(
        button_frame,
        text="NSLookup",
        bg="#8e44ad",
        fg="white",
        command=on_nslookup,
        **button_style
    )
    nslookup_button.grid(row=0, column=2, padx=5)

    netstat_button = tk.Button(
        button_frame,
        text="Netstat",
        bg="#e67e22",
        fg="white",
        command=on_netstat,
        **button_style
    )
    netstat_button.grid(row=0, column=3, padx=5)

    clear_button = tk.Button(
        button_frame,
        text="Clear Output",
        bg="#c0392b",
        fg="white",
        command=on_clear,
        **button_style
    )
    clear_button.grid(row=0, column=4, padx=5)

    status_bar = tk.Label(
        root,
        text="Ready. Enter a hostname or IP and select a diagnostic tool.",
        bd=1,
        relief=tk.SUNKEN,
        anchor=tk.W,
        font=("Helvetica", 9),
        bg="#bdc3c7"
    )
    status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    root.mainloop()


if __name__ == "__main__":
    build_gui()