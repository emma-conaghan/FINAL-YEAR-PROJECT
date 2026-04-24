import tkinter as tk
from tkinter import scrolledtext
import subprocess
import platform
import threading

def execute_network_command(command_list, display_widget):
    display_widget.delete(1.0, tk.END)
    display_widget.insert(tk.END, f"Executing: {' '.join(command_list)}\n" + "-"*50 + "\n")
    
    def work():
        try:
            # Popen allows real-time output capture from the system process
            process = subprocess.Popen(
                command_list,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            if process.stdout:
                for line in process.stdout:
                    display_widget.insert(tk.END, line)
                    display_widget.see(tk.END)
            
            process.wait()
            display_widget.insert(tk.END, "\n" + "-"*50 + "\nProcess Finished.")
        except Exception as e:
            display_widget.insert(tk.END, f"\nError occurred: {str(e)}")

    thread = threading.Thread(target=work, daemon=True)
    thread.start()

def ping_host():
    host = entry_target.get().strip()
    if not host:
        return
    # Ping flag differs between Windows (-n) and Unix-based (-c) systems
    flag = "-n" if platform.system().lower() == "windows" else "-c"
    execute_network_command(["ping", flag, "4", host], text_output)

def trace_host():
    host = entry_target.get().strip()
    if not host:
        return
    # Command differs between Windows (tracert) and Unix (traceroute)
    cmd = "tracert" if platform.system().lower() == "windows" else "traceroute"
    execute_network_command([cmd, host], text_output)

# Application UI Layout
root = tk.Tk()
root.title("Server Admin Dashboard - Network Utilities")
root.geometry("800x600")

frame_top = tk.Frame(root, pady=20)
frame_top.pack()

tk.Label(frame_top, text="Target IP or Hostname:", font=("Arial", 10)).grid(row=0, column=0, padx=5)
entry_target = tk.Entry(frame_top, width=40, font=("Arial", 10))
entry_target.grid(row=0, column=1, padx=5)

frame_btns = tk.Frame(root)
frame_btns.pack(pady=10)

btn_ping = tk.Button(frame_btns, text="Run Ping", width=20, command=ping_host, bg="#d1d1d1")
btn_ping.pack(side=tk.LEFT, padx=10)

btn_trace = tk.Button(frame_btns, text="Run Traceroute", width=20, command=trace_host, bg="#d1d1d1")
btn_trace.pack(side=tk.LEFT, padx=10)

text_output = scrolledtext.ScrolledText(root, width=90, height=25, font=("Courier New", 10), bg="#ffffff")
text_output.pack(padx=20, pady=20)

if __name__ == "__main__":
    root.mainloop()