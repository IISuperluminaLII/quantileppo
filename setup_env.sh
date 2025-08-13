
#!/usr/bin/env bash
set -e
# Use the current Python to run the cross-platform bootstrapper
python3 bootstrap_env.py || python bootstrap_env.py
