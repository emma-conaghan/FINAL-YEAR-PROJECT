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


def run_in_thread(command, output_widget, run_button):
    run_button.config(state=tk.DISABLED)

    def task():
        run_command(command, output_widget)
        run_button.config(state=tk.NORMAL)

    thread = threading.Thread(target=task, daemon=True)
    thread.start()


def get_ping_command(host, count):
    os_name = platform.system().lower()
    if os_name == "windows":
        return ["ping", "-n", str(count), host]
    else:
        return ["ping", "-c", str(count), host]


def get_traceroute_command(host):
    os_name = platform.system().lower()
    if os_name == "windows":
        return ["tracert", host]
    else:
        return ["traceroute", host]


def get_nslookup_command(host):
    return ["nslookup", host]


def get_curl_command(host):
    return ["curl", "-I", "--max-time", "10", host]


def on_ping(host_entry, count_spinbox, output_widget, run_button):
    host = host_entry.get().strip()
    if not host:
        output_widget.config(state=tk.NORMAL)
        output_widget.delete(1.0, tk.END)
        output_widget.insert(tk.END, "[Error] Please enter a hostname or IP address.")
        output_widget.config(state=tk.DISABLED)
        return
    try:
        count = int(count_spinbox.get())
    except ValueError:
        count = 4
    command = get_ping_command(host, count)
    run_in_thread(command, output_widget, run_button)


def on_traceroute(host_entry, output_widget, run_button):
    host = host_entry.get().strip()
    if not host:
        output_widget.config(state=tk.NORMAL)
        output_widget.delete(1.0, tk.END)
        output_widget.insert(tk.END, "[Error] Please enter a hostname or IP address.")
        output_widget.config(state=tk.DISABLED)
        return
    command = get_traceroute_command(host)
    run_in_thread(command, output_widget, run_button)


def on_nslookup(host_entry, output_widget, run_button):
    host = host_entry.get().strip()
    if not host:
        output_widget.config(state=tk.NORMAL)
        output_widget.delete(1.0, tk.END)
        output_widget.insert(tk.END, "[Error] Please enter a hostname or IP address.")
        output_widget.config(state=tk.DISABLED)
        return
    command = get_nslookup_command(host)
    run_in_thread(command, output_widget, run_button)


def on_curl(host_entry, output_widget, run_button):
    host = host_entry.get().strip()
    if not host:
        output_widget.config(state=tk.NORMAL)
        output_widget.delete(1.0, tk.END)
        output_widget.insert(tk.END, "[Error] Please enter a hostname or IP address.")
        output_widget.config(state=tk.DISABLED)
        return
    command = get_curl_command(host)
    run_in_thread(command, output_widget, run_button)


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
        text="Network Diagnostic Tool",
        font=("Helvetica", 16, "bold"),
        bg="#2c3e50",
        fg="white",
        pady=10
    )
    title_label.pack(fill=tk.X)

    input_frame = ttk.LabelFrame(root, text="Target", padding=10)
    input_frame.pack(fill=tk.X, padx=10, pady=5)

    tk.Label(input_frame, text="Hostname / IP:").grid(row=0, column=0, sticky=tk.W, padx=5)
    host_entry = ttk.Entry(input_frame, width=40)
    host_entry.grid(row=0, column=1, padx=5, sticky=tk.W)
    host_entry.insert(0, "google.com")

    tk.Label(input_frame, text="Ping Count:").grid(row=0, column=2, sticky=tk.W, padx=5)
    count_spinbox = ttk.Spinbox(input_frame, from_=1, to=20, width=5)
    count_spinbox.set(4)
    count_spinbox.grid(row=0, column=3, padx=5, sticky=tk.W)

    buttons_frame = ttk.LabelFrame(root, text="Actions", padding=10)
    buttons_frame.pack(fill=tk.X, padx=10, pady=5)

    output_frame = ttk.LabelFrame(root, text="Output", padding=10)
    output_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    output_widget = scrolledtext.ScrolledText(
        output_frame,
        wrap=tk.WORD,
        font=("Courier", 10),
        bg="#1e1e1e",
        fg="#00ff00",
        insertbackground="white",
        state=tk.DISABLED
    )
    output_widget.pack(fill=tk.BOTH, expand=True)

    run_button_ref = [None]

    ping_btn = ttk.Button(
        buttons_frame,
        text="Ping",
        command=lambda: on_ping(host_entry, count_spinbox, output_widget, run_button_ref[0])
    )
    ping_btn.grid(row=0, column=0, padx=5)

    traceroute_btn = ttk.Button(
        buttons_frame,
        text="Traceroute",
        command=lambda: on_traceroute(host_entry, output_widget, run_button_ref[0])
    )
    traceroute_btn.grid(row=0, column=1, padx=5)

    nslookup_btn = ttk.Button(
        buttons_frame,
        text="NSLookup",
        command=lambda: on_nslookup(host_entry, output_widget, run_button_ref[0])
    )
    nslookup_btn.grid(row=0, column=2, padx=5)

    curl_btn = ttk.Button(
        buttons_frame,
        text="HTTP Headers (curl)",
        command=lambda: on_curl(host_entry, output_widget, run_button_ref[0])
    )
    curl_btn.grid(row=0, column=3, padx=5)

    clear_btn = ttk.Button(
        buttons_frame,
        text="Clear Output",
        command=lambda: clear_output(output_widget)
    )
    clear_btn.grid(row=0, column=4, padx=5)

    run_button_ref[0] = ping_btn

    status_bar = tk.Label(
        root,
        text=f"OS: {platform.system()} {platform.release()}  |  Python {platform.python_version()}",
        bd=1,
        relief=tk.SUNKEN,
        anchor=tk.W,
        font=("Helvetica", 9),
        bg="#ecf0f1"
    )
    status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    root.mainloop()


if __name__ == "__main__":
    build_ui()