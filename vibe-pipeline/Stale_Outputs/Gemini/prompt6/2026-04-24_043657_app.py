import tkinter as tk
from tkinter import scrolledtext
import subprocess
import platform

def execute_diagnostic(command_type):
    target = host_entry.get().strip()
    if not target:
        output_area.insert(tk.END, "Error: Please enter a hostname or IP address.\n")
        return

    output_area.insert(tk.END, f"--- Running {command_type} on {target} ---\n")
    output_area.update_idletasks()

    os_type = platform.system().lower()
    
    if command_type == "Ping":
        if os_type == "windows":
            cmd = ["ping", "-n", "4", target]
        else:
            cmd = ["ping", "-c", "4", target]
    else:
        if os_type == "windows":
            cmd = ["tracert", target]
        else:
            cmd = ["traceroute", target]

    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            output_area.insert(tk.END, line)
            output_area.see(tk.END)
            output_area.update_idletasks()
        process.wait()
    except Exception as e:
        output_area.insert(tk.END, f"Execution error: {str(e)}\n")
    
    output_area.insert(tk.END, "\n--- Check Finished ---\n\n")
    output_area.see(tk.END)

def clear_logs():
    output_area.delete('1.0', tk.END)

root = tk.Tk()
root.title("Server Admin Network Utility")
root.geometry("700x500")

header_frame = tk.Frame(root, pady=10)
header_frame.pack(side=tk.TOP, fill=tk.X)

tk.Label(header_frame, text="Target Hostname/IP:").pack(side=tk.LEFT, padx=5)
host_entry = tk.Entry(header_frame, width=30)
host_entry.pack(side=tk.LEFT, padx=5)

ping_btn = tk.Button(header_frame, text="Run Ping", command=lambda: execute_diagnostic("Ping"))
ping_btn.pack(side=tk.LEFT, padx=5)

trace_btn = tk.Button(header_frame, text="Run Traceroute", command=lambda: execute_diagnostic("Traceroute"))
trace_btn.pack(side=tk.LEFT, padx=5)

clear_btn = tk.Button(header_frame, text="Clear Logs", command=clear_logs)
clear_btn.pack(side=tk.RIGHT, padx=5)

output_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, bg="black", fg="lightgreen", font=("Courier", 10))
output_area.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

root.mainloop()