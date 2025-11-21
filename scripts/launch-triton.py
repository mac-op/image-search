#!/usr/bin/env python3
import os
import subprocess
import sys
import time

# time.sleep(2)

cmd = [
    "tritonserver",
    f"--model-repository=./model-repo",
    "--model-control-mode=poll",
    "--repository-poll-secs=3",      # hot-reload every 3 seconds
    "--log-verbose=1",
    "--allow-grpc=True",
    "--allow-http=True",
    "--allow-metrics=True"
]

print("Starting Triton with hot-reload...")
print(" ".join(cmd))
subprocess.run(cmd)