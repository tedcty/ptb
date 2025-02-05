from shutil import copyfile
import os

if __name__ == '__main__':
    """
    This script builds a wheel and put it in the dist folder
    The dist folder will always have the latest build
    """
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
