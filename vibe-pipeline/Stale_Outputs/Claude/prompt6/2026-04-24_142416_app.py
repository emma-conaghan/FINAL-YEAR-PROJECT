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


def get_curl_command(host):
    return ["curl", "-I", "--max-time", "10", host]


def run_in_thread(command, output_widget, run_button):
    run_button.config(state=tk.DISABLED)

    def task():
        run_command(command, output_widget)
        run_button.config(state=tk.NORMAL)

    thread = threading.Thread(target=task, daemon=True)
    thread.start()


def on_run(host_entry, check_var, output_widget, run_button):
    host = host_entry.get().strip()
    if not host:
        output_widget.config(state=tk.NORMAL)
        output_widget.delete(1.0, tk.END)
        output_widget.insert(tk.END, "[Error] Please enter a hostname or IP address.")
        output_widget.config(state=tk.DISABLED)
        return

    check = check_var.get()

    if check == "Ping":
        command = get_ping_command(host)
    elif check == "Traceroute":
        command = get_traceroute_command(host)
    elif check == "NSLookup":
        command = get_nslookup_command(host)
    elif check == "Curl Headers":
        command = get_curl_command(host)
    else:
        return

    run_in_thread(command, output_widget, run_button)


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

    input_frame = tk.Frame(root, pady=10, padx=10)
    input_frame.pack(fill=tk.X)

    host_label = tk.Label(input_frame, text="Hostname / IP Address:", font=("Helvetica", 11))
    host_label.grid(row=0, column=0, sticky=tk.W, padx=5)

    host_entry = tk.Entry(input_frame, width=40, font=("Helvetica", 11))
    host_entry.grid(row=0, column=1, padx=5)
    host_entry.insert(0, "google.com")

    check_label = tk.Label(input_frame, text="Diagnostic Check:", font=("Helvetica", 11))
    check_label.grid(row=0, column=2, padx=5, sticky=tk.W)

    check_var = tk.StringVar(value="Ping")
    check_options = ["Ping", "Traceroute", "NSLookup", "Curl Headers"]
    check_menu = ttk.Combobox(
        input_frame,
        textvariable=check_var,
        values=check_options,
        state="readonly",
        width=15,
        font=("Helvetica", 11)
    )
    check_menu.grid(row=0, column=3, padx=5)

    output_frame = tk.Frame(root, padx=10, pady=5)
    output_frame.pack(fill=tk.BOTH, expand=True)

    output_label = tk.Label(output_frame, text="Output:", font=("Helvetica", 11, "bold"), anchor=tk.W)
    output_label.pack(anchor=tk.W)

    output_widget = scrolledtext.ScrolledText(
        output_frame,
        wrap=tk.WORD,
        font=("Courier", 10),
        bg="#1e1e1e",
        fg="#d4d4d4",
        insertbackground="white",
        state=tk.DISABLED,
        height=25
    )
    output_widget.pack(fill=tk.BOTH, expand=True)

    button_frame = tk.Frame(root, pady=10)
    button_frame.pack()

    run_button = tk.Button(
        button_frame,
        text="Run Diagnostic",
        font=("Helvetica", 11, "bold"),
        bg="#27ae60",
        fg="white",
        padx=20,
        pady=5,
        relief=tk.FLAT,
        cursor="hand2"
    )
    run_button.config(
        command=lambda: on_run(host_entry, check_var, output_widget, run_button)
    )
    run_button.pack(side=tk.LEFT, padx=10)

    clear_button = tk.Button(
        button_frame,
        text="Clear Output",
        font=("Helvetica", 11),
        bg="#e74c3c",
        fg="white",
        padx=20,
        pady=5,
        relief=tk.FLAT,
        cursor="hand2",
        command=lambda: [
            output_widget.config(state=tk.NORMAL),
            output_widget.delete(1.0, tk.END),
            output_widget.config(state=tk.DISABLED)
        ]
    )
    clear_button.pack(side=tk.LEFT, padx=10)

    status_bar = tk.Label(
        root,
        text="Ready",
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