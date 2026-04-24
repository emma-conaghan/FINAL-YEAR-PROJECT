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
        output_widget.insert(tk.END, f"\nProcess exited with code {process.returncode}\n")
    except FileNotFoundError:
        output_widget.insert(tk.END, "Error: Command not found on this system.\n")
    except Exception as e:
        output_widget.insert(tk.END, f"Error: {e}\n")

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

    input_frame = tk.Frame(root, padx=10, pady=10, bg="#ecf0f1")
    input_frame.pack(fill=tk.X)

    tk.Label(
        input_frame,
        text="Hostname / IP Address:",
        font=("Helvetica", 11),
        bg="#ecf0f1"
    ).grid(row=0, column=0, sticky=tk.W, padx=5)

    host_entry = tk.Entry(input_frame, width=40, font=("Helvetica", 11))
    host_entry.grid(row=0, column=1, padx=5)
    host_entry.insert(0, "google.com")

    button_frame = tk.Frame(root, padx=10, pady=5, bg="#ecf0f1")
    button_frame.pack(fill=tk.X)

    output_area = scrolledtext.ScrolledText(
        root,
        wrap=tk.WORD,
        font=("Courier", 10),
        state=tk.DISABLED,
        bg="#1e1e1e",
        fg="#dcdcdc",
        insertbackground="white",
        padx=10,
        pady=10
    )
    output_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    status_bar = tk.Label(
        root,
        text="Ready",
        relief=tk.SUNKEN,
        anchor=tk.W,
        font=("Helvetica", 9),
        bg="#bdc3c7"
    )
    status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def get_host():
        return host_entry.get().strip()

    def do_ping():
        host = get_host()
        if not host:
            output_area.config(state=tk.NORMAL)
            output_area.delete(1.0, tk.END)
            output_area.insert(tk.END, "Please enter a hostname or IP address.\n")
            output_area.config(state=tk.DISABLED)
            return
        status_bar.config(text=f"Running ping on {host}...")
        start_command_thread(get_ping_command(host), output_area, ping_btn)

    def do_traceroute():
        host = get_host()
        if not host:
            output_area.config(state=tk.NORMAL)
            output_area.delete(1.0, tk.END)
            output_area.insert(tk.END, "Please enter a hostname or IP address.\n")
            output_area.config(state=tk.DISABLED)
            return
        status_bar.config(text=f"Running traceroute on {host}...")
        start_command_thread(get_traceroute_command(host), output_area, traceroute_btn)

    def do_nslookup():
        host = get_host()
        if not host:
            output_area.config(state=tk.NORMAL)
            output_area.delete(1.0, tk.END)
            output_area.insert(tk.END, "Please enter a hostname or IP address.\n")
            output_area.config(state=tk.DISABLED)
            return
        status_bar.config(text=f"Running nslookup on {host}...")
        start_command_thread(get_nslookup_command(host), output_area, nslookup_btn)

    def do_whois():
        host = get_host()
        if not host:
            output_area.config(state=tk.NORMAL)
            output_area.delete(1.0, tk.END)
            output_area.insert(tk.END, "Please enter a hostname or IP address.\n")
            output_area.config(state=tk.DISABLED)
            return
        status_bar.config(text=f"Running whois on {host}...")
        start_command_thread(get_whois_command(host), output_area, whois_btn)

    def clear_output():
        output_area.config(state=tk.NORMAL)
        output_area.delete(1.0, tk.END)
        output_area.config(state=tk.DISABLED)
        status_bar.config(text="Ready")

    button_style = {"font": ("Helvetica", 10, "bold"), "width": 14, "pady": 4}

    ping_btn = tk.Button(
        button_frame,
        text="Ping",
        command=do_ping,
        bg="#2980b9",
        fg="white",
        activebackground="#1a6fa0",
        **button_style
    )
    ping_btn.grid(row=0, column=0, padx=5)

    traceroute_btn = tk.Button(
        button_frame,
        text="Traceroute",
        command=do_traceroute,
        bg="#8e44ad",
        fg="white",
        activebackground="#6c3483",
        **button_style
    )
    traceroute_btn.grid(row=0, column=1, padx=5)

    nslookup_btn = tk.Button(
        button_frame,
        text="NS Lookup",
        command=do_nslookup,
        bg="#27ae60",
        fg="white",
        activebackground="#1e8449",
        **button_style
    )
    nslookup_btn.grid(row=0, column=2, padx=5)

    whois_btn = tk.Button(
        button_frame,
        text="Whois",
        command=do_whois,
        bg="#e67e22",
        fg="white",
        activebackground="#ca6f1e",
        **button_style
    )
    whois_btn.grid(row=0, column=3, padx=5)

    clear_btn = tk.Button(
        button_frame,
        text="Clear Output",
        command=clear_output,
        bg="#c0392b",
        fg="white",
        activebackground="#a93226",
        **button_style
    )
    clear_btn.grid(row=0, column=4, padx=5)

    root.mainloop()


if __name__ == "__main__":
    build_gui()