import subprocess

from _bootstrap import add_project_root_to_path


def main():
    add_project_root_to_path()
    subprocess.check_call(["python", "-m", "onlyuav.train"])
    subprocess.check_call(["python", "-m", "onlyuav.eval"])


if __name__ == "__main__":
    main()
