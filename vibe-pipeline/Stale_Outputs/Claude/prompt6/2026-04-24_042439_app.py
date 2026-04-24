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
        output_widget.insert(tk.END, f"\n[Process finished with return code {process.returncode}]")
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


def get_curl_command(host):
    return ["curl", "-I", "--max-time", "10", host]


def on_run(action, host_var, output_widget, run_button, count_var):
    host = host_var.get().strip()
    if not host:
        output_widget.config(state=tk.NORMAL)
        output_widget.delete(1.0, tk.END)
        output_widget.insert(tk.END, "[Error] Please enter a hostname or IP address.")
        output_widget.config(state=tk.DISABLED)
        return

    try:
        count = int(count_var.get())
        if count < 1:
            count = 4
    except ValueError:
        count = 4

    if action == "ping":
        command = get_ping_command(host, count)
    elif action == "traceroute":
        command = get_traceroute_command(host)
    elif action == "nslookup":
        command = get_nslookup_command(host)
    elif action == "curl":
        command = get_curl_command(host)
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

    root.columnconfigure(0, weight=1)
    root.rowconfigure(1, weight=1)

    top_frame = ttk.LabelFrame(root, text="Diagnostic Settings", padding=10)
    top_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
    top_frame.columnconfigure(1, weight=1)

    ttk.Label(top_frame, text="Hostname / IP:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
    host_var = tk.StringVar(value="google.com")
    host_entry = ttk.Entry(top_frame, textvariable=host_var, width=40)
    host_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)

    ttk.Label(top_frame, text="Ping Count:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
    count_var = tk.StringVar(value="4")
    count_entry = ttk.Entry(top_frame, textvariable=count_var, width=10)
    count_entry.grid(row=1, column=1, sticky="w", padx=5, pady=5)

    button_frame = ttk.Frame(top_frame)
    button_frame.grid(row=2, column=0, columnspan=2, pady=10)

    run_button = ttk.Button(button_frame, text="Ping")
    ping_btn = ttk.Button(
        button_frame,
        text="Ping",
        command=lambda: on_run("ping", host_var, output_area, run_button, count_var)
    )
    ping_btn.grid(row=0, column=0, padx=5)

    traceroute_btn = ttk.Button(
        button_frame,
        text="Traceroute",
        command=lambda: on_run("traceroute", host_var, output_area, traceroute_btn, count_var)
    )
    traceroute_btn.grid(row=0, column=1, padx=5)

    nslookup_btn = ttk.Button(
        button_frame,
        text="NSLookup",
        command=lambda: on_run("nslookup", host_var, output_area, nslookup_btn, count_var)
    )
    nslookup_btn.grid(row=0, column=2, padx=5)

    curl_btn = ttk.Button(
        button_frame,
        text="HTTP Headers (curl)",
        command=lambda: on_run("curl", host_var, output_area, curl_btn, count_var)
    )
    curl_btn.grid(row=0, column=3, padx=5)

    clear_btn = ttk.Button(
        button_frame,
        text="Clear Output",
        command=lambda: clear_output(output_area)
    )
    clear_btn.grid(row=0, column=4, padx=5)

    run_button.destroy()

    output_frame = ttk.LabelFrame(root, text="Output", padding=10)
    output_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
    output_frame.columnconfigure(0, weight=1)
    output_frame.rowconfigure(0, weight=1)

    output_area = scrolledtext.ScrolledText(
        output_frame,
        wrap=tk.WORD,
        font=("Courier New", 10),
        state=tk.DISABLED,
        bg="#1e1e1e",
        fg="#d4d4d4",
        insertbackground="white"
    )
    output_area.grid(row=0, column=0, sticky="nsew")

    status_var = tk.StringVar(value="Ready")
    status_bar = ttk.Label(root, textvariable=status_var, relief=tk.SUNKEN, anchor="w")
    status_bar.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 5))

    root.mainloop()


if __name__ == "__main__":
    build_ui()