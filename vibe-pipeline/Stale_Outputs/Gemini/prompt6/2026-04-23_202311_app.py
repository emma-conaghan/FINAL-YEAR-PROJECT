import tkinter as tk
from tkinter import scrolledtext
import subprocess
import platform
import threading

def execute_command(command_list):
    try:
        process = subprocess.Popen(
            command_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        for line in process.stdout:
            output_area.insert(tk.END, line)
            output_area.see(tk.END)
            root.update_idletasks()
        process.stdout.close()
        process.wait()
    except FileNotFoundError:
        output_area.insert(tk.END, "Error: Command utility not found on this system.\n")
    except Exception as e:
        output_area.insert(tk.END, f"Execution error: {str(e)}\n")
    output_area.insert(tk.END, "\n--- Diagnostic Complete ---\n")
    output_area.see(tk.END)

def run_diagnostic(command_type):
    target = entry.get().strip()
    if not target:
        output_area.insert(tk.END, "Error: Please enter a hostname or IP address.\n")
        return

    os_name = platform.system().lower()
    if command_type == "ping":
        cmd = ["ping", "-n", "4", target] if os_name == "windows" else ["ping", "-c", "4", target]
    else:
        cmd = ["tracert", target] if os_name == "windows" else ["traceroute", target]

    output_area.insert(tk.END, f"\n--- Initiating {command_type} for {target} ---\n")
    threading.Thread(target=execute_command, args=(cmd,), daemon=True).start()

def clear_logs():
    output_area.delete('1.0', tk.END)

root = tk.Tk()
root.title("Server Admin Diagnostic Dashboard")
root.geometry("700x550")

header_label = tk.Label(root, text="Network Diagnostic Utility", font=("Arial", 14, "bold"))
header_label.pack(pady=10)

input_frame = tk.Frame(root)
input_frame.pack(pady=5)

tk.Label(input_frame, text="Target Host/IP:").pack(side=tk.LEFT, padx=5)
entry = tk.Entry(input_frame, width=40)
entry.pack(side=tk.LEFT, padx=5)
entry.insert(0, "8.8.8.8")

button_frame = tk.Frame(root)
button_frame.pack(pady=10)

ping_btn = tk.Button(button_frame, text="Run Ping", command=lambda: run_diagnostic("ping"), width=12, bg="#e1e1e1")
ping_btn.pack(side=tk.LEFT, padx=5)

trace_btn = tk.Button(button_frame, text="Run Traceroute", command=lambda: run_diagnostic("traceroute"), width=12, bg="#e1e1e1")
trace_btn.pack(side=tk.LEFT, padx=5)

clear_btn = tk.Button(button_frame, text="Clear Logs", command=clear_logs, width=12)
clear_btn.pack(side=tk.LEFT, padx=5)

output_area = scrolledtext.ScrolledText(root, width=85, height=22, bg="#1e1e1e", fg="#00ff00", font=("Consolas", 10))
output_area.pack(pady=10, padx=15)

footer = tk.Label(root, text="Note: Traceroute may require sudo/admin privileges on some systems.", font=("Arial", 8, "italic"))
footer.pack(side=tk.BOTTOM, pady=5)

if __name__ == "__main__":
    root.mainloop()