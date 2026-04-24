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
        output_widget.insert(tk.END, f"Error: Command not found. Make sure the tool is installed.\n")
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

    top_frame = tk.Frame(root, padx=10, pady=10)
    top_frame.pack(fill=tk.X)

    tk.Label(top_frame, text="Hostname / IP Address:", font=("Arial", 11)).pack(side=tk.LEFT)
    host_entry = tk.Entry(top_frame, width=35, font=("Arial", 11))
    host_entry.insert(0, "google.com")
    host_entry.pack(side=tk.LEFT, padx=8)

    ping_count_label = tk.Label(top_frame, text="Ping Count:", font=("Arial", 11))
    ping_count_label.pack(side=tk.LEFT)
    ping_count_var = tk.StringVar(value="4")
    ping_count_entry = tk.Entry(top_frame, textvariable=ping_count_var, width=5, font=("Arial", 11))
    ping_count_entry.pack(side=tk.LEFT, padx=5)

    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    def make_tab(label):
        frame = tk.Frame(notebook)
        notebook.add(frame, text=label)
        output = scrolledtext.ScrolledText(
            frame,
            wrap=tk.WORD,
            font=("Courier New", 10),
            state=tk.DISABLED,
            bg="#1e1e1e",
            fg="#d4d4d4",
            insertbackground="white"
        )
        output.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        return output

    ping_output = make_tab("Ping")
    traceroute_output = make_tab("Traceroute")
    nslookup_output = make_tab("NSLookup")
    whois_output = make_tab("Whois")

    button_frame = tk.Frame(root, padx=10, pady=8)
    button_frame.pack(fill=tk.X)

    ping_btn = tk.Button(
        button_frame,
        text="Run Ping",
        width=14,
        font=("Arial", 10, "bold"),
        bg="#0078d4",
        fg="white",
        relief=tk.FLAT
    )

    traceroute_btn = tk.Button(
        button_frame,
        text="Run Traceroute",
        width=16,
        font=("Arial", 10, "bold"),
        bg="#107c10",
        fg="white",
        relief=tk.FLAT
    )

    nslookup_btn = tk.Button(
        button_frame,
        text="Run NSLookup",
        width=15,
        font=("Arial", 10, "bold"),
        bg="#5c2d91",
        fg="white",
        relief=tk.FLAT
    )

    whois_btn = tk.Button(
        button_frame,
        text="Run Whois",
        width=13,
        font=("Arial", 10, "bold"),
        bg="#d83b01",
        fg="white",
        relief=tk.FLAT
    )

    def on_ping():
        host = host_entry.get().strip()
        if not host:
            return
        try:
            count = int(ping_count_var.get())
        except ValueError:
            count = 4
        notebook.select(0)
        cmd = get_ping_command(host, count)
        start_command_thread(cmd, ping_output, ping_btn)

    def on_traceroute():
        host = host_entry.get().strip()
        if not host:
            return
        notebook.select(1)
        cmd = get_traceroute_command(host)
        start_command_thread(cmd, traceroute_output, traceroute_btn)

    def on_nslookup():
        host = host_entry.get().strip()
        if not host:
            return
        notebook.select(2)
        cmd = get_nslookup_command(host)
        start_command_thread(cmd, nslookup_output, nslookup_btn)

    def on_whois():
        host = host_entry.get().strip()
        if not host:
            return
        notebook.select(3)
        cmd = get_whois_command(host)
        start_command_thread(cmd, whois_output, whois_btn)

    ping_btn.config(command=on_ping)
    traceroute_btn.config(command=on_traceroute)
    nslookup_btn.config(command=on_nslookup)
    whois_btn.config(command=on_whois)

    ping_btn.pack(side=tk.LEFT, padx=6)
    traceroute_btn.pack(side=tk.LEFT, padx=6)
    nslookup_btn.pack(side=tk.LEFT, padx=6)
    whois_btn.pack(side=tk.LEFT, padx=6)

    clear_btn = tk.Button(
        button_frame,
        text="Clear Output",
        width=13,
        font=("Arial", 10),
        bg="#555555",
        fg="white",
        relief=tk.FLAT,
        command=lambda: clear_current_tab(notebook, [ping_output, traceroute_output, nslookup_output, whois_output])
    )
    clear_btn.pack(side=tk.RIGHT, padx=6)

    status_bar = tk.Label(root, text="Ready", anchor=tk.W, font=("Arial", 9), bd=1, relief=tk.SUNKEN)
    status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    root.mainloop()


def clear_current_tab(notebook, outputs):
    index = notebook.index(notebook.select())
    if 0 <= index < len(outputs):
        widget = outputs[index]
        widget.config(state=tk.NORMAL)
        widget.delete(1.0, tk.END)
        widget.config(state=tk.DISABLED)


if __name__ == "__main__":
    build_gui()