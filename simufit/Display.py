import sys
import scipy.stats
from simufit.Dataset import Dataset
from itertools import chain
import simufit.dist_generator as dg
import numpy as np

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QWidget, QApplication, QMainWindow, QAction, QFileDialog
from PyQt5.QtGui import QPalette
from PyQt5.QtWidgets import QSizePolicy as qsp

class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_facecolor((53/255, 53/255, 53/255))
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor((53/255, 53/255, 53/255))
        super(MplCanvas, self).__init__(self.fig)

class Histogram(QMainWindow):

    def __init__(self, samples, bins=10, comparison_distribution=None):
        super(Histogram, self).__init__()
        self.setWindowTitle('Histogram')
        self.samples = samples
        self.comparison_distribution = comparison_distribution
        self.initUI()
        self.plotData(bins=bins)

    def initUI(self):
        """Sets up all the UI functionality."""

        ### Menu and Toolbars ###

        self.sc = MplCanvas(self, width=8, height=6, dpi=100)
        self.sc.setSizePolicy(qsp.Fixed, qsp.Fixed)

        # Create toolbar, passing canvas as first parameter, parent (self, the MainWindow) as second.
        toolbar = NavigationToolbar(self.sc, self)

        # Create grid layout for selecting distribution
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(toolbar)
        layout.addItem(QtWidgets.QSpacerItem(0, 15, qsp.Expanding, qsp.Fixed))
        layout.addItem(QtWidgets.QSpacerItem(0, 15, qsp.Expanding, qsp.Fixed))
        layout.addWidget(self.sc)

        # Create a placeholder widget to hold our toolbar and canvas.
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def plotData(self, bins):
        self.sc.axes.clear()
        alpha = 1.0

        if self.comparison_distribution is not None:
            alpha = 0.5
            self.sc.axes.hist(self.comparison_distribution.getSamples(), label=str(self.comparison_distribution._type).replace("DistributionType.","").title(), bins=bins, color="#3498DB", ec='white', alpha=alpha)
        self.sc.axes.hist(self.samples, bins=bins, label="Samples", color="#27AE60", ec='white', alpha=alpha)
        self.sc.axes.legend()
        self.sc.fig.canvas.draw_idle()


class Fitter(QMainWindow):

    def __init__(self, samples, dist=None):
        super(Fitter, self).__init__()
        self.setWindowTitle('Distribution Fitter')
        self.datasetWindow = Dataset(self)
        self.samples = samples
        self.dist = dist
        self.initUI()
        self.changeDist()
        if self.samples is not None:
            self.plotData()

    def initUI(self):
        """Sets up all the UI functionality."""

        # Set up status bar
        self.statusBar()
        self.statusBar().setStyleSheet("color: red; font-size: 14pt")

        ### Menu and Toolbars ###
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        actionImport = QAction('Import data', self)
        actionImport.setShortcut(QtGui.QKeySequence('Ctrl+I'))
        actionImport.setStatusTip('Import a data file')
        actionImport.triggered.connect(self.importData)
        fileMenu.addAction(actionImport)

        # MPL plot
        canvasHBox = QtWidgets.QHBoxLayout()
        canvasHBox.addItem(QtWidgets.QSpacerItem(0, 0, qsp.Fixed, qsp.Fixed))
        self.sc = MplCanvas(self, width=10, height=7.5, dpi=100)
        self.sc.setSizePolicy(qsp.Fixed, qsp.Fixed)
        canvasHBox.addWidget(self.sc)
        canvasHBox.addItem(QtWidgets.QSpacerItem(0, 0, qsp.Expanding, qsp.Fixed))

        # Create toolbar, passing canvas as first parameter, parent (self, the MainWindow) as second.
        toolbar = NavigationToolbar(self.sc, self)
        self.autoFitButton = QtWidgets.QPushButton('Auto Fit')
        self.autoFitButton.pressed.connect(self.autoFit)
        toolbar.addWidget(self.autoFitButton)

        # Create grid layout for selecting distribution
        self.distSelector = QtWidgets.QComboBox()
        self.distSelector.setFixedSize(QtCore.QSize(150, 40))
        if self.dist is None or self.dist.name == 'Unknown':
            dists = ['Bernoulli', 'Binomial', 'Geometric', 'Uniform', 'Normal', 'Exponential', 'Gamma', 'Weibull']
            for dist in dists:
                self.distSelector.addItem(dist)
        else:
            self.distSelector.addItem(self.dist.name)
        self.distSelector.currentTextChanged.connect(self.changeDist)

        # Slider 1
        self.slider1Label = QtWidgets.QLabel('')
        self.slider1 = QtWidgets.QSlider(minimum=1, orientation=QtCore.Qt.Horizontal, maximum=999)
        self.slider1Value = QtWidgets.QLabel('')
        self.slider1.valueChanged[int].connect(self.plotData)
        self.slider1.setFixedWidth(700)

        # Slider 2
        self.slider2Label = QtWidgets.QLabel('')
        self.slider2 = QtWidgets.QSlider(minimum=-999, orientation=QtCore.Qt.Horizontal, maximum=999)
        self.slider2.setValue(0)
        self.slider2Value = QtWidgets.QLabel('')
        self.slider2.valueChanged[int].connect(self.plotData)
        self.slider2.setFixedWidth(700)

        # Slider 3
        self.slider3Label = QtWidgets.QLabel('')
        self.slider3 = QtWidgets.QSlider(minimum=1, orientation=QtCore.Qt.Horizontal, maximum=999)
        self.slider3Value = QtWidgets.QLabel('')
        self.slider3.valueChanged[int].connect(self.plotData)
        self.slider3.setFixedWidth(700)

        # Distribution selector
        dist = QtWidgets.QGridLayout()
        dist.addWidget(QtWidgets.QLabel('Distribution:'), 0, 0)
        dist.addWidget(self.distSelector, 0, 1)
        dist.addItem(QtWidgets.QSpacerItem(0, 0, qsp.Expanding, qsp.Fixed), 0, 2)

        # Group slider labels, display values, spacers, and sliders themselves into a single layout
        self.sliders1 = QtWidgets.QHBoxLayout()
        self.sliders1.addWidget(self.slider1Label)
        self.sliders1.addItem(QtWidgets.QSpacerItem(10, 0, qsp.Fixed, qsp.Fixed))
        self.sliders1.addWidget(self.slider1Value)
        self.sliders1.addItem(QtWidgets.QSpacerItem(10, 0, qsp.Fixed, qsp.Fixed))
        self.sliders1.addWidget(self.slider1)
        self.sliders1.addItem(QtWidgets.QSpacerItem(0, 0, qsp.Expanding, qsp.Fixed))

        self.sliders2 = QtWidgets.QHBoxLayout()
        self.sliders2.addWidget(self.slider2Label)
        self.sliders2.addItem(QtWidgets.QSpacerItem(15, 0, qsp.Fixed, qsp.Fixed))
        self.sliders2.addWidget(self.slider2Value)
        self.sliders2.addItem(QtWidgets.QSpacerItem(15, 0, qsp.Fixed, qsp.Fixed))
        self.sliders2.addWidget(self.slider2)
        self.sliders2.addItem(QtWidgets.QSpacerItem(0, 0, qsp.Expanding, qsp.Fixed))

        self.sliders3 = QtWidgets.QHBoxLayout()
        self.sliders3.addWidget(self.slider3Label)
        self.sliders3.addItem(QtWidgets.QSpacerItem(15, 0, qsp.Fixed, qsp.Fixed))
        self.sliders3.addWidget(self.slider3Value)
        self.sliders3.addItem(QtWidgets.QSpacerItem(15, 0, qsp.Fixed, qsp.Fixed))
        self.sliders3.addWidget(self.slider3)
        self.sliders3.addItem(QtWidgets.QSpacerItem(0, 0, qsp.Expanding, qsp.Fixed))

        # Create final layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(toolbar)
        layout.addLayout(dist)
        layout.addItem(QtWidgets.QSpacerItem(0, 15, qsp.Expanding, qsp.Fixed))
        layout.addLayout(self.sliders1)
        layout.addLayout(self.sliders2)
        layout.addItem(QtWidgets.QSpacerItem(0, 15, qsp.Expanding, qsp.Fixed))
        layout.addLayout(self.sliders3)
        layout.addLayout(canvasHBox)

        # Create a placeholder widget to hold our toolbar and canvas.
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def autoFit(self):
        """Fit distribution to data using MLE. Display chi-square value."""

        try:
            if self.dist.name == 'Binomial':
                mle_params = self.dist.MLE(self.samples, self.slider2.value())
                self.slider1.setValue(int(mle_params * 1000))
                self.slider1Value.setText(str(round(mle_params[0], 3)))
                chisq0, chisq = self.dist.GOF(self.samples, self.slider2.value(), mle_params)
            else:
                mle_params = self.dist.MLE(self.samples)
                if self.dist.name == 'Bernoulli' or self.dist.name == 'Geometric':
                    self.slider1.setValue(int(mle_params * 1000))
                    self.slider1Value.setText(str(round(mle_params[0], 3)))
                elif self.dist.name == 'Uniform':
                    self.slider1.setValue(int(np.floor(mle_params[0])))
                    self.slider2.setValue(int(np.ceil(mle_params[1])))
                    self.slider1Value.setText(str(int(np.floor(mle_params[0]))))
                    self.slider2Value.setText(str(int(np.ceil(mle_params[1]))))
                elif self.dist.name == 'Normal':
                    self.slider2.setValue(int(mle_params[0] * 100))
                    self.slider3.setValue(int(mle_params[1] * 100))
                    self.slider2Value.setText(str(round(mle_params[0], 2)))
                    self.slider3Value.setText(str(round(mle_params[1], 2)))
                elif self.dist.name == 'Exponential':
                    self.slider1.setValue(int(mle_params * 10))
                    self.slider1Value.setText(str(round(mle_params[0], 1)))
                elif self.dist.name == 'Gamma' or self.dist.name == 'Weibull':
                    self.slider2.setValue(int(mle_params[0] * 10))
                    self.slider3.setValue(int(mle_params[1] * 10))
                    self.slider2Value.setText(str(round(mle_params[0], 3)))
                    self.slider3Value.setText(str(round(mle_params[1], 3)))
                chisq0, chisq = self.dist.GOF(self.samples, *mle_params)
            if chisq0 < chisq:
                self.statusBar().showMessage(f'Accept fit with χ0^2 = {round(chisq0, 3)} < χ^2 = {round(chisq, 3)}', 10000)
            else:
                self.statusBar().showMessage(f'Reject fit with χ0^2 = {round(chisq0, 3)} > χ^2 = {round(chisq, 3)}', 10000)
        except Exception as e:
            if self.dist.name == 'Bernoulli':
                self.statusBar().showMessage('No goodness of fit test for Bernoulli.')
            else:
                self.statusBar().showMessage(f'Goodness-of-fit test failed!.', 10000)
                print(e)

    def importData(self):
        """Import a data file (csv, txt) and view as histogram."""

        self.datasetWindow.show()

    def changeDist(self):
        """Sets up the appropriate parameter sliders for a given distribution."""

        # Checks whether samples generated from a particular known distribtion.
        # This will be the case when fit method is called from a distribution class object.
        dist = self.distSelector.currentText()

        if dist == 'Bernoulli':
            self.dist = dg.Bernoulli()
        elif dist == 'Binomial':
            self.dist = dg.Binomial()
        elif dist == 'Geometric':
            self.dist = dg.Geometric()
        elif dist == 'Uniform':
            self.dist = dg.Uniform()
        elif dist == 'Normal':
            self.dist = dg.Normal()
        elif dist == 'Exponential':
            self.dist = dg.Exponential()
        elif dist == 'Gamma':
            self.dist = dg.Gamma()
        elif dist == 'Weibull':
            self.dist = dg.Weibull()

        # Group widgets associated with a particular slider, excluding QSpacerItems
        slider1Widgets = (self.sliders1.itemAt(i).widget() for i in range(self.sliders1.count()) if not isinstance(self.sliders1.itemAt(i), QtWidgets.QSpacerItem))
        slider2Widgets = (self.sliders2.itemAt(i).widget() for i in range(self.sliders2.count()) if not isinstance(self.sliders2.itemAt(i), QtWidgets.QSpacerItem))
        slider3Widgets = (self.sliders3.itemAt(i).widget() for i in range(self.sliders3.count()) if not isinstance(self.sliders3.itemAt(i), QtWidgets.QSpacerItem))

        # Determine which widgets are visible/hidden and their labels/ranges
        if dist in ['Bernoulli', 'Geometric', 'Exponential']:
            for w in slider1Widgets:
                w.show()
            for w in chain(slider2Widgets, slider3Widgets):
                w.hide()
            self.slider1.setRange(1, 999)
            if dist in ['Bernoulli', 'Geometric']:
                self.slider1Label.setText('p')
            else:
                self.slider1Label.setText('λ')

        elif dist == 'Uniform':
            for w in chain(slider1Widgets, slider2Widgets):
                w.show()
            for w in slider3Widgets:
                w.hide()
            self.slider1Label.setText('a')
            self.slider2Label.setText('b')
            self.slider1.setRange(-999, 999)
            self.slider2.setRange(-999, 999)
            self.slider1.setValue(0)
            self.slider2.setValue(1)

        elif dist == 'Binomial':
            for w in chain(slider1Widgets, slider2Widgets):
                w.show()
            for w in slider3Widgets:
                w.hide()
            self.slider1Label.setText('p')
            self.slider2Label.setText('n')
            self.slider1.setRange(1, 999)
            self.slider2.setRange(0, 100)

        elif dist in ['Normal', 'Gamma', 'Weibull']:
            for w in slider1Widgets:
                w.hide()
            for w in chain(slider2Widgets, slider3Widgets):
                w.show()
            if dist == 'Normal':
                self.slider2Label.setText('Mean')
                self.slider3Label.setText('Variance')
                self.slider2.setRange(-9999, 9999)
                self.slider3.setRange(1, 9999)
            else:
                self.slider2Label.setText('a')
                self.slider3Label.setText('b')
                self.slider2.setRange(1, 999)
                self.slider3.setRange(1, 999)

        QtCore.QTimer.singleShot(10, lambda: self.resize(self.minimumSize()))

    def plotData(self):
        """Plots distibution PMF/PDF for a given set of parameters alongside sample data."""

        # Clear the previous plot
        self.sc.axes.clear()

        # Get distribution type
        dist = self.distSelector.currentText()

        # Plot the sample data as vertical line plot (discrete data) or histogram (continuous data).
        if dist == 'Bernoulli':
            n = len(self.samples)
            x, y = np.unique(self.samples, return_counts=True)
            self.sc.axes.scatter(x, y/n, alpha=0.6, s=150, color='lightskyblue', ec='white', label='Data')
            self.sc.axes.vlines(x, ymin=0, ymax=y/n)
        elif dist in ['Binomial', 'Geometric']:
            n = len(self.samples)
            x, y = np.unique(self.samples, return_counts=True)
            self.sc.axes.scatter(x, y/n, alpha=0.6, s=150, color='lightskyblue', ec='white', label='Data')
            self.sc.axes.vlines(x, ymin=0, ymax=y/n)
            self.sc.axes.set_xticks(np.arange(1, np.max(self.samples)+1))
        else:
            self.sc.axes.hist(self.samples, bins=np.histogram_bin_edges(self.samples, 'fd'), density=True, color=(152/255, 200/255, 132/255), ec='white', label='Data')

        x = ""
        f = ""

        # Plot the PMF/PDF of the fitted distribution
        if dist == 'Bernoulli':
            p = self.slider1.value() / 1000
            x = np.arange(2)
            y = np.array([(1 - p), p])
            self.sc.axes.scatter(x, y, alpha=0.6, s=150, color='mistyrose', ec='k', label='Fit')
            self.sc.axes.vlines(x, ymin=[0, 0], ymax=y)
            self.slider1Value.setText(str(round(p, 3)))
        elif dist == 'Binomial':
            p = self.slider1.value() / 1000
            n = self.slider2.value()
            x = np.arange(np.max(self.samples)+1)
            f = scipy.stats.binom.pmf(k=x, n=n, p=p)
            self.slider1Value.setText(str(round(p, 3)))
            self.slider2Value.setText(str(n))
        elif dist == 'Geometric':
            x = np.arange(1, np.max(self.samples)+1)
            p = self.slider1.value() / 1000
            f = (((1 - p) ** (x - 1)) * p)
            self.slider1Value.setText(str(round(p, 3)))
        elif dist == 'Uniform':
            self.slider2.setValue(np.max([self.slider1.value(), self.slider2.value()]))  # Ensure that b never goes below a
            a = self.slider1.value()
            b = self.slider2.value()
            x = np.linspace(scipy.stats.uniform.ppf(0.001, loc=a, scale=b-a), scipy.stats.uniform.ppf(0.999, loc=a, scale=b-a), 100)
            f = scipy.stats.uniform.pdf(x, loc=a, scale=b-a)
            self.slider1Value.setText(str(self.slider1.value()))
            self.slider2Value.setText(str(self.slider2.value()))
        elif dist == 'Normal':
            mean = self.slider2.value() / 100
            var = self.slider3.value() / 100
            std = np.sqrt(var)
            x = np.linspace(scipy.stats.norm.ppf(0.001, loc=mean, scale=std), scipy.stats.norm.ppf(0.999, loc=mean, scale=std), 100)
            f = scipy.stats.norm.pdf(x, loc=mean, scale=std)
            self.slider2Value.setText(str(round(mean, 3)))
            self.slider3Value.setText(str(round(var, 3)))
        elif dist == 'Exponential':
            lambd = self.slider1.value() / 10
            x = np.linspace(scipy.stats.expon.ppf(0.001, scale=1/lambd), scipy.stats.expon.ppf(0.999, scale=1/lambd), 100)
            f = scipy.stats.expon.pdf(x, scale=1/lambd)
            self.slider1Value.setText(str(round(lambd, 3)))
        elif dist == 'Gamma':
            a = self.slider2.value() / 10
            b = self.slider3.value() / 10
            x = np.linspace(scipy.stats.gamma.ppf(0.001, a, scale=b), scipy.stats.gamma.ppf(0.999, a, scale=b), 100)
            f = scipy.stats.gamma.pdf(x, a, scale=b)
            self.slider2Value.setText(str(round(a, 3)))
            self.slider3Value.setText(str(round(b, 3)))
        elif dist == 'Weibull':
            a = self.slider2.value() / 10
            b = self.slider3.value() / 10
            x = np.linspace(scipy.stats.weibull_min.ppf(0.001, a, scale=b), scipy.stats.weibull_min.ppf(0.999, a, scale=b), 100)
            f = scipy.stats.weibull_min.pdf(x, a, scale=b)
            self.slider2Value.setText(str(round(a, 3)))
            self.slider3Value.setText(str(round(b, 3)))

        if dist != 'Bernoulli':
            self.sc.axes.plot(x, f, label='Fit')
        self.sc.axes.set_ylim(bottom=0)
        self.sc.axes.set_ylabel('Frequency', color='lightskyblue', fontsize=18, labelpad=14)
        self.sc.axes.set_xlabel('x', color='lightskyblue', fontsize=18)
        self.sc.axes.tick_params(axis='both', labelsize=18, labelcolor='lightskyblue', color='lightskyblue')
        self.sc.axes.legend(fontsize=16)
        self.sc.fig.tight_layout()
        self.sc.fig.canvas.draw()

def show_histogram(samples, bins=None, comparison_distribution=None):
    if not QApplication.instance():
        app = QApplication(sys.argv)
    else:
        app = QApplication.instance()

    # Set the color scheme
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QtGui.QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, QtCore.Qt.white)
    palette.setColor(QPalette.Base, QtGui.QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QtGui.QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, QtCore.Qt.white)
    palette.setColor(QPalette.ToolTipText, QtCore.Qt.white)
    palette.setColor(QPalette.Text, QtCore.Qt.cyan)
    palette.setColor(QPalette.Button, QtGui.QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, QtCore.Qt.white)
    palette.setColor(QPalette.BrightText, QtCore.Qt.red)
    palette.setColor(QPalette.Link, QtGui.QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QtGui.QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, QtCore.Qt.black)
    app.setPalette(palette)

    window = Histogram(samples, bins, comparison_distribution)
    window.show()
    QtWidgets.QApplication.setQuitOnLastWindowClosed(True)
    app.exec_()
    app.quit()

def run_fitter(samples=None, dist=None):
    if not QApplication.instance():
        app = QApplication(sys.argv)
    else:
        app = QApplication.instance()

    # Set the color scheme
    app.setStyle("Fusion")
    app.setStyleSheet("QComboBox{font-size: 14pt} QSpinBox{font-size: 14pt} QCheckBox{font-size: 14pt} QPushButton{font-size: 14pt} QLabel{font-size: 14pt;}")
    palette = QPalette()
    palette.setColor(QPalette.Window, QtGui.QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, QtCore.Qt.white)
    palette.setColor(QPalette.Base, QtGui.QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QtGui.QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, QtCore.Qt.white)
    palette.setColor(QPalette.ToolTipText, QtCore.Qt.white)
    palette.setColor(QPalette.Text, QtCore.Qt.cyan)
    palette.setColor(QPalette.Button, QtGui.QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, QtCore.Qt.white)
    palette.setColor(QPalette.BrightText, QtCore.Qt.red)
    palette.setColor(QPalette.Link, QtGui.QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QtGui.QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, QtCore.Qt.black)
    app.setPalette(palette)

    window = Fitter(samples, dist)
    window.show()
    QtWidgets.QApplication.setQuitOnLastWindowClosed(True)
    app.exec_()
    app.quit()

class Display():
    def fit(self, samples):
        """Run the PyQt/MPL visualization."""
        run_fitter(samples, self)

    def histogram(self, samples, bins=None, comparison_distribution=None):
        """Displays the histogram of a given collection of samples, optionally a separate distribution can
        be passed to show a comparison"""
        if comparison_distribution is not None:
            if len(samples) != len(comparison_distribution.getSamples()):
                print("Distribution sample sizes do not match")
                return
        show_histogram(samples, bins, comparison_distribution)
