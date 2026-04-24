import tkinter as tk
from tkinter import scrolledtext
import subprocess
import platform
import threading

def run_diagnostic():
    target = entry.get().strip()
    if not target:
        output_area.insert(tk.END, "Error: Please enter a hostname or IP address.\n")
        return
        
    command_type = var.get()
    output_area.delete('1.0', tk.END)
    output_area.insert(tk.END, f"Running {command_type} on {target}...\n")
    
    def execute_command():
        is_windows = platform.system().lower() == "windows"
        
        if command_type == "Ping":
            cmd = ["ping", "-n", "4", target] if is_windows else ["ping", "-c", "4", target]
        else:
            cmd = ["tracert", target] if is_windows else ["traceroute", target]
        
        try:
            # shell=True is sometimes needed on Windows for built-in commands, 
            # but list-style avoids most injection and works with Popen.
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True,
                shell=False
            )
            stdout, stderr = process.communicate()
            
            def update_ui():
                if stdout:
                    output_area.insert(tk.END, stdout)
                if stderr:
                    output_area.insert(tk.END, "\nERRORS:\n" + stderr)
                output_area.insert(tk.END, "\n--- Task Finished ---\n")
                output_area.see(tk.END)
            
            root.after(0, update_ui)
            
        except FileNotFoundError:
            msg = f"Error: Command '{cmd[0]}' not found. Ensure it is installed in your PATH."
            root.after(0, lambda: output_area.insert(tk.END, msg))
        except Exception as e:
            root.after(0, lambda: output_area.insert(tk.END, f"Exception occurred: {str(e)}"))

    threading.Thread(target=execute_command, daemon=True).start()

root = tk.Tk()
root.title("Server Admin Network Dashboard")
root.geometry("700x500")

header_frame = tk.Frame(root)
header_frame.pack(pady=10)

tk.Label(header_frame, text="Target Hostname/IP:", font=("Arial", 10, "bold")).grid(row=0, column=0, padx=5)
entry = tk.Entry(header_frame, width=40)
entry.grid(row=0, column=1, padx=5)
entry.insert(0, "8.8.8.8")

options_frame = tk.Frame(root)
options_frame.pack(pady=5)

var = tk.StringVar(value="Ping")
tk.Radiobutton(options_frame, text="Ping", variable=var, value="Ping").pack(side=tk.LEFT, padx=10)
tk.Radiobutton(options_frame, text="Traceroute", variable=var, value="Traceroute").pack(side=tk.LEFT, padx=10)

btn_run = tk.Button(root, text="Run Diagnostic", command=run_diagnostic, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
btn_run.pack(pady=10)

output_area = scrolledtext.ScrolledText(root, width=80, height=20, font=("Courier New", 10))
output_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    root.mainloop()