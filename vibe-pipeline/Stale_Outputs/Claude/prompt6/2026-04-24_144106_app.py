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


def on_ping(host_entry, output_widget):
    host = host_entry.get().strip()
    if not host:
        show_error("Please enter a hostname or IP address.", output_widget)
        return
    command = get_ping_command(host)
    start_command_thread(command, output_widget)


def on_traceroute(host_entry, output_widget):
    host = host_entry.get().strip()
    if not host:
        show_error("Please enter a hostname or IP address.", output_widget)
        return
    command = get_traceroute_command(host)
    start_command_thread(command, output_widget)


def on_nslookup(host_entry, output_widget):
    host = host_entry.get().strip()
    if not host:
        show_error("Please enter a hostname or IP address.", output_widget)
        return
    command = get_nslookup_command(host)
    start_command_thread(command, output_widget)


def on_whois(host_entry, output_widget):
    host = host_entry.get().strip()
    if not host:
        show_error("Please enter a hostname or IP address.", output_widget)
        return
    command = get_whois_command(host)
    start_command_thread(command, output_widget)


def on_clear(output_widget):
    output_widget.config(state=tk.NORMAL)
    output_widget.delete(1.0, tk.END)
    output_widget.config(state=tk.DISABLED)


def show_error(message, output_widget):
    output_widget.config(state=tk.NORMAL)
    output_widget.delete(1.0, tk.END)
    output_widget.insert(tk.END, f"[Error] {message}")
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
        text="Network Diagnostic Tool",
        font=("Helvetica", 16, "bold"),
        bg="#2c3e50",
        fg="white",
        pady=10
    )
    title_label.pack(fill=tk.X)

    input_frame = tk.Frame(root, pady=8, padx=10, bg="#ecf0f1")
    input_frame.pack(fill=tk.X)

    host_label = tk.Label(input_frame, text="Hostname / IP:", font=("Helvetica", 11), bg="#ecf0f1")
    host_label.grid(row=0, column=0, padx=(0, 5))

    host_entry = ttk.Entry(input_frame, width=35, font=("Helvetica", 11))
    host_entry.grid(row=0, column=1, padx=(0, 10))
    host_entry.insert(0, "8.8.8.8")

    button_frame = tk.Frame(root, pady=5, padx=10, bg="#bdc3c7")
    button_frame.pack(fill=tk.X)

    output_area = scrolledtext.ScrolledText(
        root,
        wrap=tk.WORD,
        font=("Courier", 10),
        bg="#1e1e1e",
        fg="#dcdcdc",
        insertbackground="white",
        state=tk.DISABLED,
        padx=8,
        pady=8
    )
    output_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))

    btn_style = {"font": ("Helvetica", 10, "bold"), "width": 12, "pady": 4}

    ping_btn = tk.Button(
        button_frame,
        text="Ping",
        bg="#27ae60",
        fg="white",
        command=lambda: on_ping(host_entry, output_area),
        **btn_style
    )
    ping_btn.grid(row=0, column=0, padx=5, pady=4)

    traceroute_btn = tk.Button(
        button_frame,
        text="Traceroute",
        bg="#2980b9",
        fg="white",
        command=lambda: on_traceroute(host_entry, output_area),
        **btn_style
    )
    traceroute_btn.grid(row=0, column=1, padx=5, pady=4)

    nslookup_btn = tk.Button(
        button_frame,
        text="NSLookup",
        bg="#8e44ad",
        fg="white",
        command=lambda: on_nslookup(host_entry, output_area),
        **btn_style
    )
    nslookup_btn.grid(row=0, column=2, padx=5, pady=4)

    whois_btn = tk.Button(
        button_frame,
        text="Whois",
        bg="#e67e22",
        fg="white",
        command=lambda: on_whois(host_entry, output_area),
        **btn_style
    )
    whois_btn.grid(row=0, column=3, padx=5, pady=4)

    clear_btn = tk.Button(
        button_frame,
        text="Clear",
        bg="#c0392b",
        fg="white",
        command=lambda: on_clear(output_area),
        **btn_style
    )
    clear_btn.grid(row=0, column=4, padx=5, pady=4)

    status_bar = tk.Label(
        root,
        text=f"Platform: {platform.system()} {platform.release()}",
        anchor=tk.W,
        font=("Helvetica", 9),
        bg="#7f8c8d",
        fg="white",
        padx=6
    )
    status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    root.mainloop()


if __name__ == "__main__":
    build_ui()