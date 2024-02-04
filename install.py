import sys
import os.path
import subprocess
import os
custom_nodes_path = os.path.dirname(os.path.abspath(__file__))

def build_pip_install_cmds(args):
    if "python_embeded" in sys.executable or "python_embedded" in sys.executable:
        return [sys.executable, '-s', '-m', 'pip', 'install'] + args
    else:
        return [sys.executable, '-m', 'pip', 'install'] + args

def ensure_package():
    if os.environ.get('COMFY_SAM_ENSURE_PACKAGES',None):
        cmds = build_pip_install_cmds(['-r', 'requirements.txt'])
        subprocess.run(cmds, cwd=custom_nodes_path)

ensure_package()
