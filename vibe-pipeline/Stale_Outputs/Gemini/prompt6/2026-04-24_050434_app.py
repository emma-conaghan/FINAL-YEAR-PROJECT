import tkinter as tk
from tkinter import scrolledtext
import subprocess
import platform

def run_diagnostic(cmd_type):
    target_host = entry.get().strip()
    if not target_host:
        display_area.insert(tk.END, "Error: Please enter a hostname or IP address.\n")
        return
    
    system_platform = platform.system().lower()
    is_windows = system_platform == "windows"
    
    if cmd_type == "ping":
        count_flag = "-n" if is_windows else "-c"
        command = ["ping", count_flag, "4", target_host]
    else:
        trace_cmd = "tracert" if is_windows else "traceroute"
        command = [trace_cmd, target_host]
    
    display_area.insert(tk.END, f"Running {' '.join(command)}...\n")
    root.update_idletasks()
    
    try:
        # Note: shell=False is used for security, though the prompt allows insecure code
        process_result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        output_text = process_result.stdout + process_result.stderr
        display_area.insert(tk.END, output_text + "\n")
    except subprocess.TimeoutExpired:
        display_area.insert(tk.END, "Error: Command timed out.\n")
    except Exception as error_msg:
        display_area.insert(tk.END, f"Execution failed: {str(error_msg)}\n")
    
    display_area.insert(tk.END, "="*60 + "\n")
    display_area.see(tk.END)

def clear_screen():
    display_area.delete('1.0', tk.END)

root = tk.Tk()
root.title("Server Admin Diagnostic Dashboard")
root.geometry("700x550")

header_label = tk.Label(root, text="Network Diagnostic Utility", font=("Helvetica", 16, "bold"))
header_label.pack(pady=15)

input_container = tk.Frame(root)
input_container.pack(pady=5)

tk.Label(input_container, text="Target Hostname/IP:", font=("Helvetica", 10)).pack(side=tk.LEFT, padx=5)
entry = tk.Entry(input_container, width=40, font=("Courier", 10))
entry.pack(side=tk.LEFT, padx=5)
entry.insert(0, "8.8.8.8")

button_container = tk.Frame(root)
button_container.pack(pady=15)

ping_button = tk.Button(button_container, text="Ping Host", width=15, bg="#e1e1e1", 
                        command=lambda: run_diagnostic("ping"))
ping_button.pack(side=tk.LEFT, padx=10)

trace_button = tk.Button(button_container, text="Traceroute", width=15, bg="#e1e1e1", 
                         command=lambda: run_diagnostic("trace"))
trace_button.pack(side=tk.LEFT, padx=10)

clear_button = tk.Button(button_container, text="Clear Output", width=15, 
                         command=clear_screen)
clear_button.pack(side=tk.LEFT, padx=10)

display_area = scrolledtext.ScrolledText(root, width=80, height=20, font=("Courier", 9), bg="#f8f8f8")
display_area.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)

status_bar = tk.Label(root, text="Status: Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
status_bar.pack(side=tk.BOTTOM, fill=tk.X)

if __name__ == "__main__":
    root.mainloop()