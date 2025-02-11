from shutil import copyfile
import os
# import json
import toml

if __name__ == '__main__':
    """
    This script builds a wheel and put it in the dist folder
    The dist folder will always have the latest build
    """

    # this part of the code automatically update version information in the core.py file
    with open('pyproject.toml', 'r') as f:
        config = toml.load(f)
    info = "class info(Enum):"
    info += "\tname = \"PTB\""
    info += "\tversion = \"0.1.38\""

    info += "\tdef __str__(self):"
    info += "\t\tret = \"{0}\".format(self.value)"
    info += "\t\treturn ret"

    f = open("./ptb/core.py", "r")
    core = f.readlines()
    f.close()

    for c in range(len(core)):
        if "class info(Enum)" in core[c]:
            # Assume the value is two lines away from class header
            print(core[c+2])
            k = core[c+2].replace(core[c+2].split('=')[1].strip(), "\"" + config['project']["version"] + "\"")
            core[c + 2] = k
            break

    with open("./ptb/core.py", "w") as outfile:
        for c in core:
            outfile.write(c)
    os.listdir("./ptb/")

    if not os.path.exists('./archive/dist/'):
        os.makedirs('./archive/dist/')

    if not os.path.exists('./dist/'):
        os.makedirs('./dist/')
    files = sorted(os.listdir("./dist"))
    print(files)
    for i in files:
        copyfile('./dist/'+i, './archive/dist/'+i)
        os.remove('./dist/'+i)

    os.system('python -m build ./')
