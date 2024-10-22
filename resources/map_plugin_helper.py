import os
import sys
import zipfile
import shutil

if __name__ == '__main__':
    print("Installing Plugins ... ")
    python_path = os.path.dirname(sys.executable)
    print("Python Path ... "+python_path+"\n")
    plugins_map = python_path+"\\Lib\\site-packages\\mapclientplugins\\"
    plugins = os.getcwd()+"\\Plugin"
    plist = os.listdir(plugins)
    for pin in plist:
        pin_r = plugins+"\\"+pin
        filename, file_extension = os.path.splitext(pin_r)
        if "zip" not in file_extension:
            continue
        out_r = plugins+"\\"
        try:
            with zipfile.ZipFile(pin_r, 'r') as zip_ref:
                zip_ref.extractall(out_r)
        except IOError:
            pass
        plug_dir = plugins + "\\"+pin
        sl = pin.index("-master")
        filename, file_extension = os.path.splitext(plug_dir)
        # Source path
        src = filename+"\\mapclientplugins\\"+pin[:sl]
        print(">> Installing: "+pin[:sl])
        # Destination path
        dest = python_path + "\\Lib\\site-packages\\mapclientplugins\\"+pin[:sl]
        try:
            destination = shutil.copytree(src, dest)
        except WindowsError:
            print(">>> Plugin: "+pin[:sl]+" already installed \n>>> Location: "+dest+" \n")
