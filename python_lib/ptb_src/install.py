"""Find latest version of package and installs it."""
import fnmatch
import os


def find_whl(pattern, path):
    found = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                found.append(os.path.join(root, name))
    return sorted(found)[-1]


if __name__ == "__main__":
    os.system("pip install {}".format(find_whl('*.whl', './dist/')))
