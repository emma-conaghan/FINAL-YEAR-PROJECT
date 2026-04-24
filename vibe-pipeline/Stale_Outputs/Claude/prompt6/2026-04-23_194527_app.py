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
        output_widget.insert(tk.END, f"\n[Process finished with exit code {process.returncode}]")
    except FileNotFoundError:
        output_widget.insert(tk.END, f"[Error: Command not found: {command[0]}]")
    except Exception as e:
        output_widget.insert(tk.END, f"[Error: {str(e)}]")

    output_widget.config(state=tk.DISABLED)


def start_command_thread(command, output_widget, run_button):
    run_button.config(state=tk.DISABLED)

    def task():
        run_command(command, output_widget)
        run_button.config(state=tk.NORMAL)

    thread = threading.Thread(target=task, daemon=True)
    thread.start()


def build_ping_command(host):
    system = platform.system().lower()
    if system == "windows":
        return ["ping", "-n", "4", host]
    else:
        return ["ping", "-c", "4", host]


def build_traceroute_command(host):
    system = platform.system().lower()
    if system == "windows":
        return ["tracert", host]
    else:
        return ["traceroute", host]


def build_nslookup_command(host):
    return ["nslookup", host]


def build_whois_command(host):
    return ["whois", host]


def on_run(host_entry, tool_var, output_widget, run_button):
    host = host_entry.get().strip()
    if not host:
        output_widget.config(state=tk.NORMAL)
        output_widget.delete(1.0, tk.END)
        output_widget.insert(tk.END, "[Error: Please enter a hostname or IP address.]")
        output_widget.config(state=tk.DISABLED)
        return

    tool = tool_var.get()

    if tool == "Ping":
        command = build_ping_command(host)
    elif tool == "Traceroute":
        command = build_traceroute_command(host)
    elif tool == "NSLookup":
        command = build_nslookup_command(host)
    elif tool == "Whois":
        command = build_whois_command(host)
    else:
        output_widget.config(state=tk.NORMAL)
        output_widget.delete(1.0, tk.END)
        output_widget.insert(tk.END, "[Error: Unknown tool selected.]")
        output_widget.config(state=tk.DISABLED)
        return

    start_command_thread(command, output_widget, run_button)


def main():
    root = tk.Tk()
    root.title("Server Dashboard - Network Diagnostics")
    root.geometry("800x600")
    root.resizable(True, True)

    style = ttk.Style()
    style.theme_use("clam")

    header_label = tk.Label(
        root,
        text="Network Diagnostic Tool",
        font=("Helvetica", 18, "bold"),
        bg="#2c3e50",
        fg="white",
        pady=10
    )
    header_label.pack(fill=tk.X)

    input_frame = ttk.Frame(root, padding=10)
    input_frame.pack(fill=tk.X, padx=10, pady=5)

    ttk.Label(input_frame, text="Hostname / IP:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
    host_entry = ttk.Entry(input_frame, width=40)
    host_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
    host_entry.insert(0, "google.com")

    ttk.Label(input_frame, text="Tool:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
    tool_var = tk.StringVar(value="Ping")
    tool_combo = ttk.Combobox(
        input_frame,
        textvariable=tool_var,
        values=["Ping", "Traceroute", "NSLookup", "Whois"],
        state="readonly",
        width=15
    )
    tool_combo.grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)

    run_button = ttk.Button(
        input_frame,
        text="Run",
        command=lambda: on_run(host_entry, tool_var, output_area, run_button)
    )
    run_button.grid(row=0, column=4, padx=10, pady=5)

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

    footer_label = tk.Label(
        root,
        text="Admin Utility v1.0 | Use responsibly",
        font=("Helvetica", 9),
        bg="#2c3e50",
        fg="#aab7b8",
        pady=4
    )
    footer_label.pack(fill=tk.X, side=tk.BOTTOM)

    root.bind("<Return>", lambda event: on_run(host_entry, tool_var, output_area, run_button))

    root.mainloop()


if __name__ == "__main__":
    main()