from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QWidget, QFileDialog
from PyQt5.QtWidgets import QSizePolicy as qsp
import numpy as np


class Dataset(QWidget):

    def __init__(self, MainWindow):
        super().__init__()
        self.setWindowTitle("Import Data")
        self.initUI()
        self.mw = MainWindow

    def initUI(self):

        self.statusBar = QtWidgets.QStatusBar()

        layout = QtWidgets.QVBoxLayout()
        gridLayout = QtWidgets.QGridLayout()
        gridLayout.addWidget(QtWidgets.QLabel('File:'), 0, 0)
        self.fileTB = QtWidgets.QTextBrowser()
        self.fileTB.setFixedSize(QtCore.QSize(400, 20))
        self.browseButton = QtWidgets.QPushButton('Browse...')
        self.browseButton.clicked.connect(self.browseDialog)
        gridLayout.addWidget(self.fileTB, 0, 1, 1, 3)
        gridLayout.addWidget(self.browseButton, 0, 4)

        gridLayout.addWidget(QtWidgets.QLabel('Delimiter:'), 1, 0)
        self.commaCB = QtWidgets.QCheckBox('Comma')
        self.commaCB.setChecked(True)
        self.semiCB = QtWidgets.QCheckBox('Semi-colon')
        self.tabCB = QtWidgets.QCheckBox('Tab')
        self.tabCB.setSizePolicy(qsp.Expanding, qsp.Fixed)
        self.buttonGroup = QtWidgets.QButtonGroup(self)
        self.buttonGroup.addButton(self.commaCB, 0)
        self.buttonGroup.addButton(self.semiCB, 1)
        self.buttonGroup.addButton(self.tabCB, 2)
        gridLayout.addWidget(self.commaCB, 1, 1)
        gridLayout.addWidget(self.semiCB, 1, 2)
        gridLayout.addWidget(self.tabCB, 1, 3)
        gridLayout.addItem(QtWidgets.QSpacerItem(0, 0, qsp.Expanding, qsp.Fixed), 1, 4)

        gridLayout.addWidget(QtWidgets.QLabel('Skip rows:'), 2, 0)
        self.skipRows = QtWidgets.QSpinBox(minimum=0)
        self.skipRows.setFixedSize(40, 24)
        gridLayout.addWidget(self.skipRows)
        gridLayout.addWidget(QtWidgets.QLabel('Use column:'), 3, 0)
        self.useCol = QtWidgets.QSpinBox(minimum=0)
        self.useCol.setFixedSize(40, 24)
        gridLayout.addWidget(self.useCol)

        self.importButton = QtWidgets.QPushButton('Import')
        self.importButton.clicked.connect(self.importData)
        gridLayout.addWidget(self.importButton, 4, 4)

        layout.addLayout(gridLayout)
        layout.addWidget(self.statusBar)
        self.setLayout(layout)

    def browseDialog(self):

        fileFilter = "Data file (*.txt *.TXT *.csv *.CSV)"

        dialog = QFileDialog()
        filepath, _ = dialog.getOpenFileName(self, 'Open File', filter=fileFilter)

        if filepath:
            self.fileTB.clear()
            self.fileTB.append(filepath)

    def importData(self):

        filepath = self.fileTB.toPlainText()

        if filepath:
            delimiter = [',', ';', '\t'][self.buttonGroup.checkedId()]
            skip_header = self.skipRows.value()
            usecols = self.useCol.value()
            self.mw.samples = np.genfromtxt(fname=filepath, delimiter=delimiter, skip_header=skip_header, usecols=usecols)

            self.mw.sc.axes.clear()
            self.mw.sc.axes.hist(self.mw.samples, bins=np.histogram_bin_edges(self.mw.samples, 'fd'), density=True, color=(152/255, 200/255, 132/255), ec='white')
            self.mw.sc.fig.canvas.draw_idle()
