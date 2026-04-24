import tkinter as tk
from tkinter import scrolledtext, ttk
import subprocess
import platform
import threading

def run_diagnostic():
    target = entry_target.get().strip()
    tool = combo_tool.get()
    
    if not target:
        output_area.insert(tk.END, "Error: Please enter a hostname or IP address.\n")
        return

    output_area.delete(1.0, tk.END)
    output_area.insert(tk.END, f"Starting {tool} for {target}...\n")
    
    def execute():
        system_os = platform.system().lower()
        
        if tool == "Ping":
            if system_os == "windows":
                cmd = ["ping", "-n", "4", target]
            else:
                cmd = ["ping", "-c", "4", target]
        elif tool == "Traceroute":
            if system_os == "windows":
                cmd = ["tracert", target]
            else:
                cmd = ["traceroute", target]
        else:
            return

        try:
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True,
                shell=False
            )
            
            for line in process.stdout:
                output_area.insert(tk.END, line)
                output_area.see(tk.END)
            
            process.wait()
        except Exception as e:
            output_area.insert(tk.END, f"\nExecution failed: {str(e)}\n")

    thread = threading.Thread(target=execute)
    thread.start()

root = tk.Tk()
root.title("Server Diagnostic Utility")
root.geometry("600x450")

frame_top = tk.Frame(root, pady=10)
frame_top.pack(fill=tk.X)

tk.Label(frame_top, text="Host/IP:").pack(side=tk.LEFT, padx=5)
entry_target = tk.Entry(frame_top, width=30)
entry_target.pack(side=tk.LEFT, padx=5)

tk.Label(frame_top, text="Tool:").pack(side=tk.LEFT, padx=5)
combo_tool = ttk.Combobox(frame_top, values=["Ping", "Traceroute"], width=10)
combo_tool.current(0)
combo_tool.pack(side=tk.LEFT, padx=5)

btn_run = tk.Button(frame_top, text="Run Diagnostic", command=run_diagnostic)
btn_run.pack(side=tk.LEFT, padx=10)

output_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, bg="black", fg="lightgreen", font=("Courier", 10))
output_area.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

if __name__ == "__main__":
    root.mainloop()