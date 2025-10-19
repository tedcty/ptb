from PySide6.QtWidgets import (QFileDialog, QComboBox)


class OpenFiles(QFileDialog):

    def __init__(self, parent=None, title="Open Files"):
        super().__init__(parent)
        self.setWindowTitle(title)
        # options = QFileDialog.Options()
        # options |= QFileDialog.Option.DontUseNativeDialog
        self.setOption(QFileDialog.Option.DontUseNativeDialog)
        #self.setOption(options)
        comb0 = self.findChild(QComboBox, "lookInCombo")
        comb0.setEditable(True)
        line0 = comb0.lineEdit()
        line0.returnPressed.connect(lambda:
                                    self.setDirectory(line0.text()))

    def get_files(self, file_filter='Text (*.txt)'):
        self.setNameFilter(file_filter)
        self.setFileMode(QFileDialog.FileMode.ExistingFiles)
        return self.__open__up__()

    def __open__up__(self):
        if self.exec():
            file_names = self.selectedFiles()
            self.close()
            return file_names
        return None

    def get_file(self, file_filter='Text (*.txt)'):
        self.setNameFilter(file_filter)
        self.setFileMode(QFileDialog.FileMode.ExistingFile)
        ret = self.__open__up__()
        if ret is not None:
            return ret[0]
        return ret

    def get_save_file(self, file_filter='Text (*.txt)'):
        self.setNameFilter(file_filter)
        self.setFileMode(QFileDialog.FileMode.AnyFile)
        self.setLabelText(QFileDialog.DialogLabel.Accept, "Save")
        ret = self.__open__up__()
        if ret is not None:
            return ret[0]
        return ret

    def get_dir(self):
        self.setFileMode(QFileDialog.FileMode.Directory)
        ret = self.__open__up__()
        if ret is not None:
            return ret[0]
        return ret

    def get_dirs(self):
        self.setFileMode(QFileDialog.FileMode.Directory)
        return self.__open__up__()

