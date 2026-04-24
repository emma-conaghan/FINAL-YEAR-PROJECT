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
            text=True,
            bufsize=1
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


def run_in_thread(command, output_widget, run_button):
    run_button.config(state=tk.DISABLED)

    def task():
        run_command(command, output_widget)
        run_button.config(state=tk.NORMAL)

    thread = threading.Thread(target=task, daemon=True)
    thread.start()


def build_ping_command(host, count):
    system = platform.system().lower()
    if system == "windows":
        return ["ping", "-n", str(count), host]
    else:
        return ["ping", "-c", str(count), host]


def build_traceroute_command(host):
    system = platform.system().lower()
    if system == "windows":
        return ["tracert", host]
    else:
        return ["traceroute", host]


def build_nslookup_command(host):
    return ["nslookup", host]


def build_nmap_command(host):
    return ["nmap", "-sV", "--open", host]


def on_run(host_entry, tool_var, ping_count_entry, output_widget, run_button):
    host = host_entry.get().strip()
    if not host:
        output_widget.config(state=tk.NORMAL)
        output_widget.delete(1.0, tk.END)
        output_widget.insert(tk.END, "Please enter a hostname or IP address.\n")
        output_widget.config(state=tk.DISABLED)
        return

    tool = tool_var.get()

    if tool == "Ping":
        try:
            count = int(ping_count_entry.get())
            if count < 1:
                count = 4
        except ValueError:
            count = 4
        command = build_ping_command(host, count)
    elif tool == "Traceroute":
        command = build_traceroute_command(host)
    elif tool == "NSLookup":
        command = build_nslookup_command(host)
    elif tool == "Nmap":
        command = build_nmap_command(host)
    else:
        return

    run_in_thread(command, output_widget, run_button)


def on_tool_change(tool_var, ping_frame):
    if tool_var.get() == "Ping":
        ping_frame.grid()
    else:
        ping_frame.grid_remove()


def main():
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

    controls_frame = tk.Frame(root, pady=10, padx=10, bg="#ecf0f1")
    controls_frame.pack(fill=tk.X)

    tk.Label(controls_frame, text="Hostname / IP:", bg="#ecf0f1", font=("Helvetica", 11)).grid(
        row=0, column=0, sticky=tk.W, padx=5, pady=5
    )
    host_entry = tk.Entry(controls_frame, width=30, font=("Helvetica", 11))
    host_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
    host_entry.insert(0, "google.com")

    tk.Label(controls_frame, text="Tool:", bg="#ecf0f1", font=("Helvetica", 11)).grid(
        row=0, column=2, sticky=tk.W, padx=5, pady=5
    )
    tool_var = tk.StringVar(value="Ping")
    tool_dropdown = ttk.Combobox(
        controls_frame,
        textvariable=tool_var,
        values=["Ping", "Traceroute", "NSLookup", "Nmap"],
        state="readonly",
        width=15,
        font=("Helvetica", 11)
    )
    tool_dropdown.grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)

    ping_frame = tk.Frame(controls_frame, bg="#ecf0f1")
    ping_frame.grid(row=0, column=4, sticky=tk.W, padx=5, pady=5)
    tk.Label(ping_frame, text="Count:", bg="#ecf0f1", font=("Helvetica", 11)).pack(side=tk.LEFT)
    ping_count_entry = tk.Entry(ping_frame, width=5, font=("Helvetica", 11))
    ping_count_entry.pack(side=tk.LEFT, padx=3)
    ping_count_entry.insert(0, "4")

    tool_dropdown.bind("<<ComboboxSelected>>", lambda e: on_tool_change(tool_var, ping_frame))

    run_button = tk.Button(
        controls_frame,
        text="Run",
        font=("Helvetica", 11, "bold"),
        bg="#27ae60",
        fg="white",
        padx=15,
        pady=5,
        relief=tk.FLAT,
        cursor="hand2"
    )
    run_button.grid(row=0, column=5, padx=10, pady=5)
    run_button.config(
        command=lambda: on_run(host_entry, tool_var, ping_count_entry, output_text, run_button)
    )

    clear_button = tk.Button(
        controls_frame,
        text="Clear",
        font=("Helvetica", 11),
        bg="#e74c3c",
        fg="white",
        padx=10,
        pady=5,
        relief=tk.FLAT,
        cursor="hand2",
        command=lambda: (
            output_text.config(state=tk.NORMAL),
            output_text.delete(1.0, tk.END),
            output_text.config(state=tk.DISABLED)
        )
    )
    clear_button.grid(row=0, column=6, padx=5, pady=5)

    info_frame = tk.Frame(root, bg="#bdc3c7", padx=10, pady=5)
    info_frame.pack(fill=tk.X)
    info_label = tk.Label(
        info_frame,
        text="Tip: Enter a valid hostname or IP, select a tool, and click Run.",
        bg="#bdc3c7",
        font=("Helvetica", 9, "italic"),
        fg="#2c3e50"
    )
    info_label.pack(anchor=tk.W)

    output_frame = tk.Frame(root, padx=10, pady=10)
    output_frame.pack(fill=tk.BOTH, expand=True)

    tk.Label(output_frame, text="Output:", font=("Helvetica", 11, "bold")).pack(anchor=tk.W)

    output_text = scrolledtext.ScrolledText(
        output_frame,
        wrap=tk.WORD,
        font=("Courier", 10),
        bg="#1e1e1e",
        fg="#dcdcdc",
        insertbackground="white",
        state=tk.DISABLED
    )
    output_text.pack(fill=tk.BOTH, expand=True)

    status_bar = tk.Label(
        root,
        text=f"System: {platform.system()} {platform.release()}",
        bd=1,
        relief=tk.SUNKEN,
        anchor=tk.W,
        font=("Helvetica", 9),
        bg="#2c3e50",
        fg="white"
    )
    status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    host_entry.bind("<Return>", lambda e: on_run(host_entry, tool_var, ping_count_entry, output_text, run_button))

    root.mainloop()


if __name__ == "__main__":
    main()