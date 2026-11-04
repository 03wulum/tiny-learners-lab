"""Run every experiment: python run_all.py [--quick | --clean [--all]]"""
import os, shutil, subprocess, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(ROOT, "experiments")


def main():
    args = sys.argv[1:]
    rc = 0
    for name in sorted(os.listdir(EXP)):
        d = os.path.join(EXP, name)
        if os.path.isfile(os.path.join(d, "run.py")):
            print(f"\n=== {name} ===", flush=True)
            rc |= subprocess.call([sys.executable, "run.py", *args], cwd=d)
    if "--clean" in args and "--all" in args:
        venv = os.path.join(ROOT, ".venv")
        if os.path.isdir(venv) and os.path.abspath(sys.prefix) != os.path.abspath(venv):
            shutil.rmtree(venv)
            print("deleted .venv")
        elif os.path.isdir(venv):
            print("Running inside .venv: deactivate it, then delete the .venv folder to finish cleanup.")
    sys.exit(rc)


if __name__ == "__main__":
    main()
