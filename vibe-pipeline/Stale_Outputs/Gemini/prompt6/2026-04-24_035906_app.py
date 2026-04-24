import tkinter as tk
from tkinter import scrolledtext
import subprocess
import platform
import threading

def execute_diagnostic(cmd, display_widget):
    def task():
        display_widget.config(state=tk.NORMAL)
        display_widget.delete(1.0, tk.END)
        display_widget.insert(tk.END, f"Running command: {cmd}\n" + "-"*50 + "\n")
        display_widget.config(state=tk.DISABLED)
        
        try:
            process = subprocess.Popen(
                cmd, 
                shell=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            for line in process.stdout:
                display_widget.config(state=tk.NORMAL)
                display_widget.insert(tk.END, line)
                display_widget.see(tk.END)
                display_widget.config(state=tk.DISABLED)
            
            process.stdout.close()
            process.wait()
        except Exception as e:
            display_widget.config(state=tk.NORMAL)
            display_widget.insert(tk.END, f"\nExecution Error: {str(e)}")
            display_widget.config(state=tk.DISABLED)
            
    thread = threading.Thread(target=task)
    thread.daemon = True
    thread.start()

def run_ping():
    target = host_entry.get().strip()
    if target:
        param = "-n" if platform.system().lower() == "windows" else "-c"
        command = f"ping {param} 4 {target}"
        execute_diagnostic(command, output_area)

def run_traceroute():
    target = host_entry.get().strip()
    if target:
        command = "tracert" if platform.system().lower() == "windows" else "traceroute"
        execute_diagnostic(f"{command} {target}", output_area)

root = tk.Tk()
root.title("Server Admin Network Utility")
root.geometry("800x600")

header = tk.Label(root, text="Network Diagnostic Dashboard", font=("Segoe UI", 16, "bold"))
header.pack(pady=15)

input_frame = tk.Frame(root)
input_frame.pack(pady=10)

tk.Label(input_frame, text="Hostname or IP Address:", font=("Segoe UI", 10)).pack(side=tk.LEFT, padx=5)
host_entry = tk.Entry(input_frame, width=40, font=("Segoe UI", 10))
host_entry.pack(side=tk.LEFT, padx=5)
host_entry.insert(0, "8.8.8.8")

button_frame = tk.Frame(root)
button_frame.pack(pady=10)

ping_btn = tk.Button(button_frame, text="Ping Test", command=run_ping, width=15, bg="#e1e1e1")
ping_btn.pack(side=tk.LEFT, padx=10)

trace_btn = tk.Button(button_frame, text="Traceroute", command=run_traceroute, width=15, bg="#e1e1e1")
trace_btn.pack(side=tk.LEFT, padx=10)

output_label = tk.Label(root, text="Console Output:", font=("Segoe UI", 10, "italic"))
output_label.pack(anchor="w", padx=20)

output_area = scrolledtext.ScrolledText(root, width=90, height=25, bg="#1e1e1e", fg="#d4d4d4", font=("Consolas", 10))
output_area.pack(pady=5, padx=20, fill=tk.BOTH, expand=True)
output_area.config(state=tk.DISABLED)

if __name__ == "__main__":
    root.mainloop()