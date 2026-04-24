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


def run_in_thread(command, output_widget, run_button):
    run_button.config(state=tk.DISABLED)
    output_widget.config(state=tk.NORMAL)
    output_widget.delete(1.0, tk.END)
    output_widget.insert(tk.END, "Please wait...\n")
    output_widget.config(state=tk.DISABLED)

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


def build_whois_command(host):
    return ["whois", host]


def on_run(host_entry, tool_var, ping_count_var, output_widget, run_button):
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
            count = int(ping_count_var.get())
            if count < 1:
                count = 4
        except ValueError:
            count = 4
        command = build_ping_command(host, count)
    elif tool == "Traceroute":
        command = build_traceroute_command(host)
    elif tool == "NSLookup":
        command = build_nslookup_command(host)
    elif tool == "Whois":
        command = build_whois_command(host)
    else:
        return

    run_in_thread(command, output_widget, run_button)


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

    control_frame = tk.Frame(root, bg="#ecf0f1", pady=10, padx=10)
    control_frame.pack(fill=tk.X)

    tk.Label(control_frame, text="Host / IP:", bg="#ecf0f1", font=("Helvetica", 11)).grid(
        row=0, column=0, sticky=tk.W, padx=5, pady=5
    )
    host_entry = tk.Entry(control_frame, width=30, font=("Helvetica", 11))
    host_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
    host_entry.insert(0, "google.com")

    tk.Label(control_frame, text="Tool:", bg="#ecf0f1", font=("Helvetica", 11)).grid(
        row=0, column=2, sticky=tk.W, padx=5, pady=5
    )
    tool_var = tk.StringVar(value="Ping")
    tool_menu = ttk.Combobox(
        control_frame,
        textvariable=tool_var,
        values=["Ping", "Traceroute", "NSLookup", "Whois"],
        state="readonly",
        width=15,
        font=("Helvetica", 11)
    )
    tool_menu.grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)

    tk.Label(control_frame, text="Ping Count:", bg="#ecf0f1", font=("Helvetica", 11)).grid(
        row=0, column=4, sticky=tk.W, padx=5, pady=5
    )
    ping_count_var = tk.StringVar(value="4")
    ping_count_entry = tk.Entry(control_frame, textvariable=ping_count_var, width=5, font=("Helvetica", 11))
    ping_count_entry.grid(row=0, column=5, sticky=tk.W, padx=5, pady=5)

    run_button = tk.Button(
        control_frame,
        text="Run",
        font=("Helvetica", 11, "bold"),
        bg="#27ae60",
        fg="white",
        padx=10,
        pady=5,
        relief=tk.FLAT,
        cursor="hand2"
    )
    run_button.grid(row=0, column=6, padx=10, pady=5)

    run_button.config(
        command=lambda: on_run(host_entry, tool_var, ping_count_var, output_text, run_button)
    )

    separator = ttk.Separator(root, orient="horizontal")
    separator.pack(fill=tk.X, pady=2)

    output_frame = tk.Frame(root, bg="#1e1e1e", padx=5, pady=5)
    output_frame.pack(fill=tk.BOTH, expand=True)

    output_label = tk.Label(
        output_frame,
        text="Output:",
        bg="#1e1e1e",
        fg="#bdc3c7",
        font=("Helvetica", 10, "bold"),
        anchor="w"
    )
    output_label.pack(fill=tk.X)

    output_text = scrolledtext.ScrolledText(
        output_frame,
        font=("Courier", 10),
        bg="#1e1e1e",
        fg="#00ff00",
        insertbackground="white",
        state=tk.DISABLED,
        wrap=tk.WORD
    )
    output_text.pack(fill=tk.BOTH, expand=True)

    clear_button = tk.Button(
        root,
        text="Clear Output",
        font=("Helvetica", 10),
        bg="#e74c3c",
        fg="white",
        relief=tk.FLAT,
        cursor="hand2",
        command=lambda: (
            output_text.config(state=tk.NORMAL),
            output_text.delete(1.0, tk.END),
            output_text.config(state=tk.DISABLED)
        )
    )
    clear_button.pack(pady=5)

    status_bar = tk.Label(
        root,
        text="Ready",
        bd=1,
        relief=tk.SUNKEN,
        anchor=tk.W,
        font=("Helvetica", 9),
        bg="#bdc3c7"
    )
    status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    root.bind("<Return>", lambda event: on_run(host_entry, tool_var, ping_count_var, output_text, run_button))

    root.mainloop()


if __name__ == "__main__":
    main()