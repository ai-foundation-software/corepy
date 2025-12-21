# Troubleshooting Installation Cancellation

If you encounter "Error: The operation was canceled" during your setup (especially after seemingly successful package installs), it likely indicates a **timeout** or a **resource exhaustion (memory)** event that terminated the process or the agent's connection to it.

## 1. Diagnose the Cause

### Check for Out-of-Memory (OOM)
Compiling C++ extensions (via `pip` and `cmake`) is memory-intensive. If the system runs out of RAM, the Linux kernel will kill the process.
Run this command to check for OOM events:
```bash
sudo dmesg | grep -i "killed"
# OR
grep -i "out of memory" /var/log/syslog
```
*If you see "Out of memory: Kill process...", this is the cause.*

### Check for Timeouts
If you are running this via an AI Agent, IDE Task, or CI job, there is often a fixed execution time limit (e.g., 60 minutes or 10 minutes).
- **Agent/IDE**: Check the settings for "Timeout" or "Execution Time Limit".
- **Logs**: Look for a hard cut-off in the logs with no specific error message from the compiler itself.

---

## 2. Safely Resume or Restart

### Cleanup Lock Files (If apt was involved)
If `apt` install was interrupted, the lock file might be stuck.
```bash
# Check if dpkg is interrupted
sudo dpkg --configure -a

# If you get lock errors:
sudo rm /var/lib/apt/lists/lock
sudo rm /var/cache/apt/archives/lock
sudo rm /var/lib/dpkg/lock*
```

### Resume Python/C++ Build
If `pip install` was cancelled, it is generally safe to re-run it. Since you are using `scikit-build-core` / `cmake`, re-running should pick up cached object files.
```bash
# Clean build artifacts if you want a fresh start (optional but recommended if stuck)
rm -rf _skbuild dist build

# Re-run installation with verbose logging to see progress
pip install --no-build-isolation -v -e .
```

---

## 3. Configuration & Prevention

### For IDE / Environment
1.  **Increase Timeouts**: If this is a VS Code Task, check `.vscode/tasks.json` and ensure there isn't a restrictive `timeout`.
2.  **Parallel Jobs**: Restrict `ninja` or `make` parallelism to save memory.
    ```bash
    # Set max jobs to 2 (or 1) to reduce memory usage
    export CMAKE_BUILD_PARALLEL_LEVEL=2
    pip install -e .
    ```

### For AI Agents
If you are asking an AI to run this:
*   **Request Async Execution**: Ask the agent to "Start the build in the background and check status later".
*   **Split Steps**: Instead of one giant install script, ask to install dependencies first, then build the code.

### Environment Ssanity Check
Ensure your tools are actually available:
```bash
cmake --version
ninja --version
python3 -c "import sys; print(sys.version)"
```
