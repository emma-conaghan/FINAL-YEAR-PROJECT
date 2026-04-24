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
    except FileNotFoundError as e:
        output_widget.insert(tk.END, f"Error: Command not found - {e}")
    except Exception as e:
        output_widget.insert(tk.END, f"Error: {e}")

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


def main():
    root = tk.Tk()
    root.title("Server Dashboard - Network Diagnostics")
    root.geometry("800x600")
    root.resizable(True, True)

    top_frame = tk.Frame(root, padx=10, pady=10)
    top_frame.pack(fill=tk.X)

    tk.Label(top_frame, text="Hostname / IP Address:", font=("Arial", 11)).pack(side=tk.LEFT)

    host_var = tk.StringVar(value="google.com")
    host_entry = tk.Entry(top_frame, textvariable=host_var, width=30, font=("Arial", 11))
    host_entry.pack(side=tk.LEFT, padx=8)

    tk.Label(top_frame, text="Ping Count:", font=("Arial", 11)).pack(side=tk.LEFT)

    count_var = tk.StringVar(value="4")
    count_spinbox = tk.Spinbox(top_frame, from_=1, to=20, textvariable=count_var, width=5, font=("Arial", 11))
    count_spinbox.pack(side=tk.LEFT, padx=8)

    button_frame = tk.Frame(root, padx=10, pady=5)
    button_frame.pack(fill=tk.X)

    output_frame = tk.Frame(root, padx=10, pady=5)
    output_frame.pack(fill=tk.BOTH, expand=True)

    tk.Label(output_frame, text="Output:", font=("Arial", 11, "bold")).pack(anchor=tk.W)

    output_area = scrolledtext.ScrolledText(
        output_frame,
        state=tk.DISABLED,
        font=("Courier", 10),
        bg="#1e1e1e",
        fg="#d4d4d4",
        insertbackground="white",
        wrap=tk.WORD
    )
    output_area.pack(fill=tk.BOTH, expand=True)

    status_var = tk.StringVar(value="Ready")
    status_bar = tk.Label(root, textvariable=status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W, font=("Arial", 9))
    status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    run_button_ref = {}

    def on_ping():
        host = host_var.get().strip()
        if not host:
            status_var.set("Please enter a hostname or IP address.")
            return
        try:
            count = int(count_var.get())
        except ValueError:
            count = 4
        command = build_ping_command(host, count)
        status_var.set(f"Running ping on {host}...")
        start_command_thread(command, output_area, run_button_ref["ping"])

    def on_traceroute():
        host = host_var.get().strip()
        if not host:
            status_var.set("Please enter a hostname or IP address.")
            return
        command = build_traceroute_command(host)
        status_var.set(f"Running traceroute on {host}...")
        start_command_thread(command, output_area, run_button_ref["traceroute"])

    def on_nslookup():
        host = host_var.get().strip()
        if not host:
            status_var.set("Please enter a hostname or IP address.")
            return
        command = build_nslookup_command(host)
        status_var.set(f"Running nslookup on {host}...")
        start_command_thread(command, output_area, run_button_ref["nslookup"])

    def on_whois():
        host = host_var.get().strip()
        if not host:
            status_var.set("Please enter a hostname or IP address.")
            return
        command = build_whois_command(host)
        status_var.set(f"Running whois on {host}...")
        start_command_thread(command, output_area, run_button_ref["whois"])

    def on_clear():
        output_area.config(state=tk.NORMAL)
        output_area.delete(1.0, tk.END)
        output_area.config(state=tk.DISABLED)
        status_var.set("Output cleared.")

    ping_btn = tk.Button(
        button_frame, text="Ping", command=on_ping,
        bg="#0078d7", fg="white", font=("Arial", 10, "bold"),
        padx=10, pady=4, relief=tk.FLAT, cursor="hand2"
    )
    ping_btn.pack(side=tk.LEFT, padx=4)
    run_button_ref["ping"] = ping_btn

    traceroute_btn = tk.Button(
        button_frame, text="Traceroute", command=on_traceroute,
        bg="#107c10", fg="white", font=("Arial", 10, "bold"),
        padx=10, pady=4, relief=tk.FLAT, cursor="hand2"
    )
    traceroute_btn.pack(side=tk.LEFT, padx=4)
    run_button_ref["traceroute"] = traceroute_btn

    nslookup_btn = tk.Button(
        button_frame, text="NSLookup", command=on_nslookup,
        bg="#8764b8", fg="white", font=("Arial", 10, "bold"),
        padx=10, pady=4, relief=tk.FLAT, cursor="hand2"
    )
    nslookup_btn.pack(side=tk.LEFT, padx=4)
    run_button_ref["nslookup"] = nslookup_btn

    whois_btn = tk.Button(
        button_frame, text="Whois", command=on_whois,
        bg="#c7611a", fg="white", font=("Arial", 10, "bold"),
        padx=10, pady=4, relief=tk.FLAT, cursor="hand2"
    )
    whois_btn.pack(side=tk.LEFT, padx=4)
    run_button_ref["whois"] = whois_btn

    clear_btn = tk.Button(
        button_frame, text="Clear Output", command=on_clear,
        bg="#d13438", fg="white", font=("Arial", 10, "bold"),
        padx=10, pady=4, relief=tk.FLAT, cursor="hand2"
    )
    clear_btn.pack(side=tk.RIGHT, padx=4)

    root.mainloop()


if __name__ == "__main__":
    main()