
# PowerShell wrapper for cross-platform bootstrapper
# Prefer 'python' on PATH; fall back to py launcher if available
$ErrorActionPreference = "Stop"
try {
    python bootstrap_env.py
} catch {
    py -3 bootstrap_env.py
}
