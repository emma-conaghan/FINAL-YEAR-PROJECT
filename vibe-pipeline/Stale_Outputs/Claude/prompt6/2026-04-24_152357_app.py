import anthropic
import subprocess
import platform
import sys

client = anthropic.Anthropic()

def run_ping(host: str) -> str:
    """Run ping command for the given host."""
    system = platform.system().lower()
    
    if system == "windows":
        cmd = ["ping", "-n", "4", host]
    else:
        cmd = ["ping", "-c", "4", host]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        output = result.stdout
        if result.stderr:
            output += "\nErrors:\n" + result.stderr
        return output
    except subprocess.TimeoutExpired:
        return f"Ping to {host} timed out after 30 seconds"
    except FileNotFoundError:
        return f"Ping command not found on this system"
    except Exception as e:
        return f"Error running ping: {str(e)}"

def run_traceroute(host: str) -> str:
    """Run traceroute command for the given host."""
    system = platform.system().lower()
    
    if system == "windows":
        cmd = ["tracert", host]
    else:
        cmd = ["traceroute", host]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        output = result.stdout
        if result.stderr:
            output += "\nErrors:\n" + result.stderr
        return output
    except subprocess.TimeoutExpired:
        return f"Traceroute to {host} timed out after 60 seconds"
    except FileNotFoundError:
        return f"Traceroute command not found on this system. Try installing it first."
    except Exception as e:
        return f"Error running traceroute: {str(e)}"

def process_tool_call(tool_name: str, tool_input: dict) -> str:
    """Process tool calls from Claude."""
    host = tool_input.get("host", "")
    
    if tool_name == "run_ping":
        print(f"\n[Running ping to {host}...]")
        result = run_ping(host)
        return result
    elif tool_name == "run_traceroute":
        print(f"\n[Running traceroute to {host}...]")
        result = run_traceroute(host)
        return result
    else:
        return f"Unknown tool: {tool_name}"

def run_network_diagnostics(host: str, diagnostic_type: str = "both"):
    """
    Run network diagnostics using Claude with tool use.
    
    Args:
        host: The hostname or IP address to diagnose
        diagnostic_type: "ping", "traceroute", or "both"
    """
    
    tools = [
        {
            "name": "run_ping",
            "description": "Run a ping command to test connectivity to a host. Returns the ping output including response times and packet loss.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "host": {
                        "type": "string",
                        "description": "The hostname or IP address to ping"
                    }
                },
                "required": ["host"]
            }
        },
        {
            "name": "run_traceroute",
            "description": "Run a traceroute command to trace the network path to a host. Returns the traceroute output showing each hop.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "host": {
                        "type": "string",
                        "description": "The hostname or IP address to trace route to"
                    }
                },
                "required": ["host"]
            }
        }
    ]
    
    if diagnostic_type == "ping":
        user_message = f"Please run a ping test to {host} and provide a detailed analysis of the results."
    elif diagnostic_type == "traceroute":
        user_message = f"Please run a traceroute to {host} and provide a detailed analysis of the network path."
    else:
        user_message = f"Please run both a ping test and a traceroute to {host}, then provide a comprehensive network diagnostic report."
    
    print(f"\n{'='*60}")
    print(f"Network Diagnostic Report for: {host}")
    print(f"{'='*60}")
    print(f"\nAnalyzing network connectivity to {host}...")
    
    messages = [{"role": "user", "content": user_message}]
    
    while True:
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=4096,
            tools=tools,
            messages=messages
        )
        
        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, 'text'):
                    print("\n" + "="*60)
                    print("ANALYSIS:")
                    print("="*60)
                    print(block.text)
            break
        
        if response.stop_reason == "tool_use":
            tool_results = []
            
            for block in response.content:
                if block.type == "tool_use":
                    tool_result = process_tool_call(block.name, block.input)
                    print(f"\nRaw output:")
                    print("-" * 40)
                    print(tool_result)
                    print("-" * 40)
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": tool_result
                    })
            
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
        else:
            print(f"Unexpected stop reason: {response.stop_reason}")
            break

def get_user_input():
    """Get host and diagnostic type from user."""
    print("\n" + "="*60)
    print("SERVER DASHBOARD - Network Diagnostic Utility")
    print("="*60)
    
    host = input("\nEnter hostname or IP address to diagnose: ").strip()
    if not host:
        print("Error: Please provide a valid hostname or IP address")
        return None, None
    
    print("\nSelect diagnostic type:")
    print("1. Ping only")
    print("2. Traceroute only")
    print("3. Both (comprehensive)")
    
    choice = input("\nEnter your choice (1-3, default=3): ").strip()
    
    if choice == "1":
        diagnostic_type = "ping"
    elif choice == "2":
        diagnostic_type = "traceroute"
    else:
        diagnostic_type = "both"
    
    return host, diagnostic_type

def main():
    """Main function to run the network diagnostic utility."""
    print("\nWelcome to the Server Dashboard Network Diagnostic Utility")
    print("This tool uses AI to analyze network connectivity and provide insights.")
    
    while True:
        host, diagnostic_type = get_user_input()
        
        if host is None:
            continue
        
        run_network_diagnostics(host, diagnostic_type)
        
        another = input("\n\nRun another diagnostic? (yes/no): ").strip().lower()
        if another not in ['yes', 'y']:
            print("\nThank you for using the Network Diagnostic Utility. Goodbye!")
            break

if __name__ == "__main__":
    main()