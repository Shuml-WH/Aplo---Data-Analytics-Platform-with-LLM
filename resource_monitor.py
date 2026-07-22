"""
Resource Monitor - Logs RAM and CPU usage for Flask and Node.js processes.
Run this script alongside the Flask app to track resource consumption.

Usage:
    python resource_monitor.py [--interval 30] [--output resource_log.txt]
"""

import time
import argparse
import datetime
import os
import threading

try:
    import psutil
except ImportError:
    print("psutil not installed. Installing...")
    os.system("pip install psutil")
    import psutil


def find_flask_process():
    """Find the Flask (Python) process running app.py."""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info', 'cpu_percent']):
        try:
            cmdline = proc.info.get('cmdline') or []
            cmdline_str = ' '.join(cmdline).lower()
            if ('python' in cmdline_str or 'flask' in cmdline_str) and 'app.py' in cmdline_str:
                return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


def find_node_process():
    """Find the Node.js process specifically for this project's Vite dev server."""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info', 'cpu_percent']):
        try:
            cmdline = proc.info.get('cmdline') or []
            cmdline_str = ' '.join(cmdline).lower()
            # Match only if it's a node process AND references this project's directory or vite
            is_node = 'node' in (proc.info.get('name') or '').lower()
            has_vite = 'vite' in cmdline_str
            has_project_path = 'aplo-dashboard' in cmdline_str or 'aplo' in cmdline_str
            if is_node and (has_vite or has_project_path):
                return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


def get_process_resources(proc, label):
    """Get RAM and CPU usage for a specific process."""
    try:
        mem = proc.memory_info()
        cpu = proc.cpu_percent(interval=0.5)
        return {
            "label": label,
            "pid": proc.pid,
            "ram_mb": round(mem.rss / (1024 ** 2), 2),
            "cpu_percent": cpu,
        }
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def get_resource_usage():
    """Get current RAM and CPU usage for Flask and Node.js processes."""
    flask = find_flask_process()
    node = find_node_process()

    flask_res = get_process_resources(flask, "Flask") if flask else None
    node_res = get_process_resources(node, "Node.js") if node else None

    return {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "flask": flask_res,
        "node": node_res,
    }


def format_log_entry(usage):
    """Format a resource usage entry as a log line."""
    flask = usage.get("flask")
    node = usage.get("node")

    parts = [f"[{usage['timestamp']}]"]

    if flask:
        parts.append(f"Flask (PID {flask['pid']}): RAM={flask['ram_mb']} MB, CPU={flask['cpu_percent']}%")
    else:
        parts.append("Flask: not found")

    if node:
        parts.append(f"Node.js (PID {node['pid']}): RAM={node['ram_mb']} MB, CPU={node['cpu_percent']}%")
    else:
        parts.append("Node.js: not found")

    return " | ".join(parts)


def monitor(interval=30, output_file="resource_log.txt"):
    """Main monitoring loop."""
    print(f"Resource Monitor started. Logging every {interval} seconds to {output_file}")
    print("Tracking: Flask (app.py) and Node.js (Vite dev server)")
    print("Press Ctrl+C to stop.\n")

    with open(output_file, "a") as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Resource Monitor started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Logging interval: {interval} seconds\n")
        f.write(f"Tracking: Flask (app.py) and Node.js (Vite dev server)\n")
        f.write(f"{'='*80}\n\n")

    try:
        while True:
            usage = get_resource_usage()
            log_line = format_log_entry(usage)

            # Print to console
            print(log_line)

            # Write to file
            with open(output_file, "a") as f:
                f.write(log_line + "\n")

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\nResource Monitor stopped.")
        with open(output_file, "a") as f:
            f.write(f"\nResource Monitor stopped at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


def start_background_monitor(interval=30, output_file="resource_log.txt"):
    """Start the resource monitor in a background daemon thread.
    Call this from Flask app.py to auto-start monitoring."""
    def _run():
        # Small delay to let Flask startup complete
        time.sleep(2)
        monitor(interval=interval, output_file=output_file)

    thread = threading.Thread(target=_run, daemon=True, name="ResourceMonitor")
    thread.start()
    print(f"[resource_monitor] Background monitor started (interval={interval}s, log={output_file})")
    return thread


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor Flask and Node.js resource usage")
    parser.add_argument("--interval", type=int, default=30, help="Logging interval in seconds (default: 30)")
    parser.add_argument("--output", type=str, default="resource_log.txt", help="Output log file (default: resource_log.txt)")
    args = parser.parse_args()

    monitor(interval=args.interval, output_file=args.output)
