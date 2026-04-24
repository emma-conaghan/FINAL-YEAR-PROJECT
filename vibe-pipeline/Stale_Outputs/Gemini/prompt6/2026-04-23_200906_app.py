import tkinter as tk
from tkinter import scrolledtext
import subprocess
import platform
import threading

def execute_network_tool(command_type):
    target = entry_host.get().strip()
    if not target:
        output_display.insert(tk.END, "Error: Please enter a hostname or IP address.\n")
        return

    output_display.insert(tk.END, f"--- Starting {command_type} on {target} ---\n")
    output_display.see(tk.END)

    def run_process():
        os_name = platform.system().lower()
        
        if command_type == "Ping":
            flag = "-n" if os_name == "windows" else "-c"
            cmd = ["ping", flag, "4", target]
        else:
            cmd_name = "tracert" if os_name == "windows" else "traceroute"
            cmd = [cmd_name, target]

        try:
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True,
                shell=(os_name == "windows")
            )
            
            for line in process.stdout:
                output_display.insert(tk.END, line)
                output_display.see(tk.END)
            
            process.wait()
            output_display.insert(tk.END, f"--- {command_type} Finished ---\n\n")
        except Exception as e:
            output_display.insert(tk.END, f"Execution Error: {str(e)}\n\n")
        
        output_display.see(tk.END)

    # Run in thread to keep GUI responsive
    threading.Thread(target=run_process, daemon=True).start()

def clear_output():
    output_display.delete('1.0', tk.END)

# UI Setup
root = tk.Tk()
root.title("Server Admin Dashboard - Network Diagnostics")
root.geometry("700x500")

# Input Frame
input_frame = tk.Frame(root)
input_frame.pack(pady=10, fill=tk.X, padx=10)

tk.Label(input_frame, text="Hostname/IP:").pack(side=tk.LEFT, padx=5)
entry_host = tk.Entry(input_frame, width=40)
entry_host.pack(side=tk.LEFT, padx=5)
entry_host.insert(0, "8.8.8.8")

# Buttons
btn_ping = tk.Button(input_frame, text="Run Ping", command=lambda: execute_network_tool("Ping"))
btn_ping.pack(side=tk.LEFT, padx=5)

btn_trace = tk.Button(input_frame, text="Run Traceroute", command=lambda: execute_network_tool("Traceroute"))
btn_trace.pack(side=tk.LEFT, padx=5)

btn_clear = tk.Button(input_frame, text="Clear Logs", command=clear_output)
btn_clear.pack(side=tk.RIGHT, padx=5)

# Output Display
output_display = scrolledtext.ScrolledText(root, wrap=tk.WORD, bg="black", fg="lightgreen", font=("Courier New", 10))
output_display.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

if __name__ == "__main__":
    root.mainloop()