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
        output_widget.insert(tk.END, "Error: Command not found on this system.\n")
    except Exception as e:
        output_widget.insert(tk.END, f"Error: {str(e)}\n")

    output_widget.config(state=tk.DISABLED)


def start_command_thread(command, output_widget, run_button):
    run_button.config(state=tk.DISABLED)

    def task():
        run_command(command, output_widget)
        run_button.config(state=tk.NORMAL)

    thread = threading.Thread(target=task, daemon=True)
    thread.start()


def get_ping_command(host):
    os_name = platform.system().lower()
    if os_name == "windows":
        return ["ping", "-n", "4", host]
    else:
        return ["ping", "-c", "4", host]


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


def run_diagnostic(action, host_entry, output_widget, run_button):
    host = host_entry.get().strip()
    if not host:
        output_widget.config(state=tk.NORMAL)
        output_widget.delete(1.0, tk.END)
        output_widget.insert(tk.END, "Please enter a hostname or IP address.\n")
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


def clear_output(output_widget):
    output_widget.config(state=tk.NORMAL)
    output_widget.delete(1.0, tk.END)
    output_widget.config(state=tk.DISABLED)


def build_ui():
    root = tk.Tk()
    root.title("Server Dashboard - Network Diagnostics")
    root.geometry("800x600")
    root.resizable(True, True)

    style = ttk.Style()
    style.theme_use("clam")

    title_label = tk.Label(
        root,
        text="Server Network Diagnostic Tool",
        font=("Helvetica", 16, "bold"),
        bg="#2c3e50",
        fg="white",
        pady=10
    )
    title_label.pack(fill=tk.X)

    input_frame = ttk.LabelFrame(root, text="Target Host", padding=10)
    input_frame.pack(fill=tk.X, padx=10, pady=5)

    host_label = ttk.Label(input_frame, text="Hostname / IP Address:")
    host_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)

    host_entry = ttk.Entry(input_frame, width=40, font=("Helvetica", 11))
    host_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
    host_entry.insert(0, "google.com")

    button_frame = ttk.LabelFrame(root, text="Diagnostics", padding=10)
    button_frame.pack(fill=tk.X, padx=10, pady=5)

    output_frame = ttk.LabelFrame(root, text="Output", padding=10)
    output_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

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

    ping_button = ttk.Button(
        button_frame,
        text="Ping",
        command=lambda: run_diagnostic("ping", host_entry, output_area, ping_button)
    )
    ping_button.grid(row=0, column=0, padx=5, pady=5)

    trace_button = ttk.Button(
        button_frame,
        text="Traceroute",
        command=lambda: run_diagnostic("traceroute", host_entry, output_area, trace_button)
    )
    trace_button.grid(row=0, column=1, padx=5, pady=5)

    nslookup_button = ttk.Button(
        button_frame,
        text="NSLookup",
        command=lambda: run_diagnostic("nslookup", host_entry, output_area, nslookup_button)
    )
    nslookup_button.grid(row=0, column=2, padx=5, pady=5)

    whois_button = ttk.Button(
        button_frame,
        text="Whois",
        command=lambda: run_diagnostic("whois", host_entry, output_area, whois_button)
    )
    whois_button.grid(row=0, column=3, padx=5, pady=5)

    clear_button = ttk.Button(
        button_frame,
        text="Clear Output",
        command=lambda: clear_output(output_area)
    )
    clear_button.grid(row=0, column=4, padx=5, pady=5)

    status_bar = tk.Label(
        root,
        text=f"System: {platform.system()} {platform.release()}",
        bd=1,
        relief=tk.SUNKEN,
        anchor=tk.W,
        font=("Helvetica", 9)
    )
    status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    root.mainloop()


if __name__ == "__main__":
    build_ui()