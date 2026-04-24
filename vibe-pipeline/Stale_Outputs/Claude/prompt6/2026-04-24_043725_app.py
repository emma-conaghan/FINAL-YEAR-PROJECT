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
        output_widget.insert(tk.END, f"Error: Command not found - {e}\n")
    except Exception as e:
        output_widget.insert(tk.END, f"Error: {e}\n")

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


def build_gui():
    root = tk.Tk()
    root.title("Server Dashboard - Network Diagnostics")
    root.geometry("900x650")
    root.resizable(True, True)

    style = ttk.Style()
    style.theme_use("clam")

    header = tk.Label(
        root,
        text="Network Diagnostic Tool",
        font=("Helvetica", 18, "bold"),
        bg="#2c3e50",
        fg="white",
        pady=10
    )
    header.pack(fill=tk.X)

    input_frame = ttk.LabelFrame(root, text="Target Host", padding=10)
    input_frame.pack(fill=tk.X, padx=10, pady=5)

    tk.Label(input_frame, text="Hostname / IP Address:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
    host_entry = ttk.Entry(input_frame, width=40, font=("Helvetica", 12))
    host_entry.grid(row=0, column=1, padx=5, pady=5)
    host_entry.insert(0, "google.com")

    tk.Label(input_frame, text="Ping Count:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
    ping_count_entry = ttk.Entry(input_frame, width=5, font=("Helvetica", 12))
    ping_count_entry.grid(row=0, column=3, padx=5, pady=5)
    ping_count_entry.insert(0, "4")

    button_frame = ttk.LabelFrame(root, text="Diagnostic Commands", padding=10)
    button_frame.pack(fill=tk.X, padx=10, pady=5)

    output_frame = ttk.LabelFrame(root, text="Output", padding=10)
    output_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    output_area = scrolledtext.ScrolledText(
        output_frame,
        wrap=tk.WORD,
        font=("Courier", 10),
        bg="#1e1e1e",
        fg="#00ff00",
        insertbackground="white",
        state=tk.DISABLED,
        height=20
    )
    output_area.pack(fill=tk.BOTH, expand=True)

    def on_ping():
        host = host_entry.get().strip()
        if not host:
            output_area.config(state=tk.NORMAL)
            output_area.delete(1.0, tk.END)
            output_area.insert(tk.END, "Please enter a hostname or IP address.\n")
            output_area.config(state=tk.DISABLED)
            return
        try:
            count = int(ping_count_entry.get().strip())
        except ValueError:
            count = 4
        command = get_ping_command(host, count)
        start_command_thread(command, output_area, ping_btn)

    def on_traceroute():
        host = host_entry.get().strip()
        if not host:
            output_area.config(state=tk.NORMAL)
            output_area.delete(1.0, tk.END)
            output_area.insert(tk.END, "Please enter a hostname or IP address.\n")
            output_area.config(state=tk.DISABLED)
            return
        command = get_traceroute_command(host)
        start_command_thread(command, output_area, traceroute_btn)

    def on_nslookup():
        host = host_entry.get().strip()
        if not host:
            output_area.config(state=tk.NORMAL)
            output_area.delete(1.0, tk.END)
            output_area.insert(tk.END, "Please enter a hostname or IP address.\n")
            output_area.config(state=tk.DISABLED)
            return
        command = get_nslookup_command(host)
        start_command_thread(command, output_area, nslookup_btn)

    def on_whois():
        host = host_entry.get().strip()
        if not host:
            output_area.config(state=tk.NORMAL)
            output_area.delete(1.0, tk.END)
            output_area.insert(tk.END, "Please enter a hostname or IP address.\n")
            output_area.config(state=tk.DISABLED)
            return
        command = get_whois_command(host)
        start_command_thread(command, output_area, whois_btn)

    def on_clear():
        output_area.config(state=tk.NORMAL)
        output_area.delete(1.0, tk.END)
        output_area.config(state=tk.DISABLED)

    ping_btn = ttk.Button(button_frame, text="Ping", command=on_ping, width=15)
    ping_btn.grid(row=0, column=0, padx=5, pady=5)

    traceroute_btn = ttk.Button(button_frame, text="Traceroute", command=on_traceroute, width=15)
    traceroute_btn.grid(row=0, column=1, padx=5, pady=5)

    nslookup_btn = ttk.Button(button_frame, text="NS Lookup", command=on_nslookup, width=15)
    nslookup_btn.grid(row=0, column=2, padx=5, pady=5)

    whois_btn = ttk.Button(button_frame, text="Whois", command=on_whois, width=15)
    whois_btn.grid(row=0, column=3, padx=5, pady=5)

    clear_btn = ttk.Button(button_frame, text="Clear Output", command=on_clear, width=15)
    clear_btn.grid(row=0, column=4, padx=5, pady=5)

    status_bar = tk.Label(
        root,
        text=f"System: {platform.system()} | Python: {platform.python_version()}",
        bd=1,
        relief=tk.SUNKEN,
        anchor=tk.W,
        font=("Helvetica", 9),
        bg="#ecf0f1"
    )
    status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    root.mainloop()


if __name__ == "__main__":
    build_gui()