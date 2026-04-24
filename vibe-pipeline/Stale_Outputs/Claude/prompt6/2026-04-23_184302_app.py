import subprocess
import platform
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox


def run_command(command):
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        output = result.stdout
        if result.stderr:
            output += "\n[STDERR]\n" + result.stderr
        return output if output.strip() else "No output returned."
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 30 seconds."
    except Exception as e:
        return f"Error running command: {str(e)}"


def get_ping_command(host, count=4):
    system = platform.system().lower()
    if system == "windows":
        return f"ping -n {count} {host}"
    else:
        return f"ping -c {count} {host}"


def get_traceroute_command(host):
    system = platform.system().lower()
    if system == "windows":
        return f"tracert {host}"
    else:
        return f"traceroute {host}"


def get_nslookup_command(host):
    return f"nslookup {host}"


def get_whois_command(host):
    system = platform.system().lower()
    if system == "windows":
        return f"whois {host}"
    else:
        return f"whois {host}"


class ServerDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Server Admin Dashboard - Network Diagnostics")
        self.root.geometry("900x700")
        self.root.configure(bg="#2b2b2b")

        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("TFrame", background="#2b2b2b")
        self.style.configure("TLabel", background="#2b2b2b", foreground="#ffffff", font=("Consolas", 11))
        self.style.configure("TButton", background="#444444", foreground="#ffffff", font=("Consolas", 10), padding=5)
        self.style.map("TButton", background=[("active", "#666666")])
        self.style.configure("TEntry", font=("Consolas", 11))
        self.style.configure("TLabelframe", background="#2b2b2b", foreground="#aaaaaa")
        self.style.configure("TLabelframe.Label", background="#2b2b2b", foreground="#aaaaaa", font=("Consolas", 10))

        self.build_ui()

    def build_ui(self):
        title_label = ttk.Label(
            self.root,
            text="Server Admin Dashboard",
            font=("Consolas", 18, "bold"),
            foreground="#00ff88"
        )
        title_label.pack(pady=(15, 5))

        subtitle_label = ttk.Label(
            self.root,
            text="Network Diagnostic Utility",
            font=("Consolas", 11),
            foreground="#888888"
        )
        subtitle_label.pack(pady=(0, 15))

        input_frame = ttk.LabelFrame(self.root, text="Target Host", padding=10)
        input_frame.pack(fill="x", padx=20, pady=5)

        host_label = ttk.Label(input_frame, text="Hostname / IP Address:")
        host_label.grid(row=0, column=0, padx=(0, 10), sticky="w")

        self.host_entry = ttk.Entry(input_frame, width=40, font=("Consolas", 11))
        self.host_entry.grid(row=0, column=1, padx=(0, 10), sticky="w")
        self.host_entry.insert(0, "google.com")

        ping_count_label = ttk.Label(input_frame, text="Ping Count:")
        ping_count_label.grid(row=0, column=2, padx=(10, 5), sticky="w")

        self.ping_count_var = tk.StringVar(value="4")
        ping_count_spinbox = tk.Spinbox(
            input_frame,
            from_=1,
            to=20,
            textvariable=self.ping_count_var,
            width=5,
            font=("Consolas", 11),
            bg="#444444",
            fg="#ffffff",
            buttonbackground="#555555"
        )
        ping_count_spinbox.grid(row=0, column=3, padx=(0, 10), sticky="w")

        buttons_frame = ttk.LabelFrame(self.root, text="Diagnostic Tools", padding=10)
        buttons_frame.pack(fill="x", padx=20, pady=5)

        self.ping_btn = ttk.Button(buttons_frame, text="Ping", command=self.run_ping, width=15)
        self.ping_btn.grid(row=0, column=0, padx=5, pady=5)

        self.traceroute_btn = ttk.Button(buttons_frame, text="Traceroute", command=self.run_traceroute, width=15)
        self.traceroute_btn.grid(row=0, column=1, padx=5, pady=5)

        self.nslookup_btn = ttk.Button(buttons_frame, text="NSLookup", command=self.run_nslookup, width=15)
        self.nslookup_btn.grid(row=0, column=2, padx=5, pady=5)

        self.whois_btn = ttk.Button(buttons_frame, text="Whois", command=self.run_whois, width=15)
        self.whois_btn.grid(row=0, column=3, padx=5, pady=5)

        self.clear_btn = ttk.Button(buttons_frame, text="Clear Output", command=self.clear_output, width=15)
        self.clear_btn.grid(row=0, column=4, padx=5, pady=5)

        output_frame = ttk.LabelFrame(self.root, text="Output", padding=10)
        output_frame.pack(fill="both", expand=True, padx=20, pady=10)

        self.output_text = scrolledtext.ScrolledText(
            output_frame,
            wrap=tk.WORD,
            font=("Consolas", 10),
            bg="#1a1a1a",
            fg="#00ff88",
            insertbackground="#ffffff",
            relief="flat",
            padx=10,
            pady=10
        )
        self.output_text.pack(fill="both", expand=True)

        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(
            self.root,
            textvariable=self.status_var,
            foreground="#888888",
            font=("Consolas", 9)
        )
        status_bar.pack(side="bottom", pady=5)

    def get_host(self):
        host = self.host_entry.get().strip()
        if not host:
            messagebox.showwarning("Input Error", "Please enter a hostname or IP address.")
            return None
        return host

    def append_output(self, text):
        self.output_text.config(state="normal")
        self.output_text.insert(tk.END, text + "\n")
        self.output_text.see(tk.END)

    def clear_output(self):
        self.output_text.config(state="normal")
        self.output_text.delete(1.0, tk.END)
        self.status_var.set("Output cleared.")

    def set_buttons_state(self, state):
        self.ping_btn.config(state=state)
        self.traceroute_btn.config(state=state)
        self.nslookup_btn.config(state=state)
        self.whois_btn.config(state=state)

    def run_diagnostic(self, command, label):
        self.set_buttons_state("disabled")
        self.status_var.set(f"Running {label}...")
        self.root.update()

        separator = "=" * 60
        self.append_output(f"\n{separator}")
        self.append_output(f"  {label}")
        self.append_output(f"  Command: {command}")
        self.append_output(f"{separator}\n")

        output = run_command(command)
        self.append_output(output)
        self.append_output(f"\n{separator}\n")

        self.status_var.set(f"{label} completed.")
        self.set_buttons_state("normal")

    def run_ping(self):
        host = self.get_host()
        if not host:
            return
        try:
            count = int(self.ping_count_var.get())
        except ValueError:
            count = 4
        command = get_ping_command(host, count)
        self.run_diagnostic(command, f"PING -> {host}")

    def run_traceroute(self):
        host = self.get_host()
        if not host:
            return
        command = get_traceroute_command(host)
        self.run_diagnostic(command, f"TRACEROUTE -> {host}")

    def run_nslookup(self):
        host = self.get_host()
        if not host:
            return
        command = get_nslookup_command(host)
        self.run_diagnostic(command, f"NSLOOKUP -> {host}")

    def run_whois(self):
        host = self.get_host()
        if not host:
            return
        command = get_whois_command(host)
        self.run_diagnostic(command, f"WHOIS -> {host}")


def main():
    root = tk.Tk()
    app = ServerDashboard(root)
    root.mainloop()


if __name__ == "__main__":
    main()