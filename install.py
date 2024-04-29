import sys
import os.path
import subprocess

custom_nodes_path = os.path.dirname(os.path.abspath(__file__))

def build_pip_install_cmds(args: list[str]) -> list[str]:
    """Constructs the appropriate pip install command based on the current Python environment, including the necessary arguments provided as input."""
    if "python_embeded" in sys.executable or "python_embedded" in sys.executable:
        return [sys.executable, '-s', '-m', 'pip', 'install'] + args
    else:
        return [sys.executable, '-m', 'pip', 'install'] + args

def ensure_package() -> None:
    """Calls the build_pip_install_cmds function to construct pip install commands and then runs the commands using subprocess in a specified directory."""
    cmds = build_pip_install_cmds(['-r', 'requirements.txt'])
    subprocess.run(cmds, cwd=custom_nodes_path)

ensure_package()