import tkinter as tk
from tkinter import scrolledtext
import subprocess
import platform

def run_command(cmd_type):
    target = entry.get().strip()
    if not target:
        output_area.insert(tk.END, "Error: Please enter a hostname or IP address.\n")
        return

    output_area.delete('1.0', tk.END)
    output_area.insert(tk.END, f"Executing {cmd_type} for {target}...\n")
    root.update_idletasks()

    os_type = platform.system().lower()
    
    if cmd_type == "Ping":
        if os_type == "windows":
            command = ["ping", "-n", "4", target]
        else:
            command = ["ping", "-c", "4", target]
    else: # Traceroute
        if os_type == "windows":
            command = ["tracert", target]
        else:
            command = ["traceroute", target]

    try:
        process = subprocess.run(command, capture_output=True, text=True, timeout=60)
        if process.stdout:
            output_area.insert(tk.END, process.stdout)
        if process.stderr:
            output_area.insert(tk.END, "\nERRORS:\n" + process.stderr)
    except subprocess.TimeoutExpired:
        output_area.insert(tk.END, "Error: Operation timed out.")
    except Exception as e:
        output_area.insert(tk.END, f"Execution error: {str(e)}")

root = tk.Tk()
root.title("Server Admin Diagnostic Utility")
root.geometry("700x500")

header_label = tk.Label(root, text="Network Diagnostics Tool", font=("Arial", 14, "bold"))
header_label.pack(pady=10)

input_frame = tk.Frame(root)
input_frame.pack(pady=5)

tk.Label(input_frame, text="Target Host/IP:").pack(side=tk.LEFT, padx=5)
entry = tk.Entry(input_frame, width=40)
entry.pack(side=tk.LEFT, padx=5)

button_frame = tk.Frame(root)
button_frame.pack(pady=10)

ping_btn = tk.Button(button_frame, text="Run Ping", width=15, command=lambda: run_command("Ping"))
ping_btn.pack(side=tk.LEFT, padx=10)

trace_btn = tk.Button(button_frame, text="Run Traceroute", width=15, command=lambda: run_command("Traceroute"))
trace_btn.pack(side=tk.LEFT, padx=10)

output_area = scrolledtext.ScrolledText(root, width=80, height=20, font=("Courier New", 10))
output_area.pack(padx=10, pady=10)

root.mainloop()