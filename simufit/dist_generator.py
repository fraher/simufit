import numpy as np
from simufit.Helpers import mergeBins, gammaMLE, weibullMLE
from scipy.optimize import minimize
import scipy.stats
import scipy.special

<<<<<<< HEAD
from simufit.Types import MeasureType as mt
from simufit.Display import Display
=======
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
                    self.slider1.setValue(int(np.round(mle_params[0], 1) * 10))
                    self.slider2.setValue(int(np.round(mle_params[1], 1) * 10))
                    self.slider1Value.setText(str(np.round(mle_params[0], 2)))
                    self.slider2Value.setText(str(np.round(mle_params[1], 2)))
                elif self.dist.name == 'Normal':
                    self.slider2.setValue(int(np.round(mle_params[0], 1) * 10))
                    self.slider3.setValue(int(np.round(mle_params[1], 1) * 10))
                    self.slider2Value.setText(str(np.round(mle_params[0], 2)))
                    self.slider3Value.setText(str(np.round(mle_params[1], 2)))
                elif self.dist.name == 'Exponential':
                    self.slider1.setValue(int(mle_params * 10))
                    self.slider1Value.setText(str(round(mle_params[0], 2)))
                elif self.dist.name == 'Gamma' or self.dist.name == 'Weibull':
                    self.slider2.setValue(int(mle_params[0] * 10))
                    self.slider3.setValue(int(mle_params[1] * 10))
                    self.slider2Value.setText(str(round(mle_params[0], 2)))
                    self.slider3Value.setText(str(round(mle_params[1], 2)))
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
            self.dist = Bernoulli()
        elif dist == 'Binomial':
            self.dist = Binomial()
        elif dist == 'Geometric':
            self.dist = Geometric()
        elif dist == 'Uniform':
            self.dist = Uniform()
        elif dist == 'Normal':
            self.dist = Normal()
        elif dist == 'Exponential':
            self.dist = Exponential()
        elif dist == 'Gamma':
            self.dist = Gamma()
        elif dist == 'Weibull':
            self.dist = Weibull()

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
                self.slider2.setRange(-999, 999)
                self.slider3.setRange(1, 999)
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
            a = self.slider1.value() / 10
            b = self.slider2.value() / 10
            x = np.linspace(scipy.stats.uniform.ppf(0.001, loc=a, scale=b-a), scipy.stats.uniform.ppf(0.999, loc=a, scale=b-a), 100)
            f = scipy.stats.uniform.pdf(x, loc=a, scale=b-a)
            self.slider1Value.setText(str(np.round(a, 1)))
            self.slider2Value.setText(str(np.round(b, 1)))
        elif dist == 'Normal':
            mean = self.slider2.value() / 10
            var = self.slider3.value() / 10
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
>>>>>>> master

class Bernoulli(Display):

    def __init__(self):
        self.name = 'Bernoulli'
        self.measure_type = mt.DISCRETE
        self._parameters = [{'label': 'p', 'probability':[0,1]}]

    def sample(self, p, size=None, seed=None):
        """Get samples from Bern(p). The size argument is the number of samples (default 1)."""

        if seed is not None:
            np.random.seed(seed)

        if p <= 0 or p >= 1:
            raise ValueError('p must be in the range (0, 1).')

        samples = np.random.uniform(size=size)
        mask1 = samples >= p
        mask2 = samples < p
        samples[mask1] = 0
        samples[mask2] = 1

        return samples.astype(np.int64)

    def negLogL(self, p, samples):
        """Calculate the negative log likelihood for a collection of random
        Bernoulli-distributed samples, and a specified p."""

        n = len(samples)
        m = np.sum(samples)

        return (m * np.log(p) + (n - m) * np.log(1 - p)) * -1

    def MLE(self, samples, use_minimizer=False, p0=None):
        """Returns the maximum likelihood estimate of parameter p, given a collection of samples.
        If use_minimizer=True, an initial guess p0 for p must be provided. Otherwise, the closed
        form expression for the MLE of p is used."""

        if use_minimizer:
            if p0 is None:
                raise ValueError('Supply an initial guess p0=p to the optimizer.')
            if p0 <= 0 or p0 >= 1:
                raise ValueError('p must be in the range (0, 1). Supply an initial guess in this range.')
            res = minimize(self.negLogL, p0, args=samples, method='Nelder-Mead')
            if res.status == 1:
                print(f'Warning: Optimizer failed to converge with initial guess {p0}. Returned None for MLE values. Try another initial guess.')
                return None
            else:
                return res.x

        else:
            return np.array([np.mean(samples)])

class Binomial(Display):

    def __init__(self):
        self.name = 'Binomial'
        self.measure_type = mt.DISCRETE
        self._parameters = [{'label': 'n', 'range':[0,100]}, {'label': 'p', 'probability':[0,1]}]

    def sample(self, n, p, size=None, seed=None):
        """Get samples from Bin(n, p). The size argument is the number of samples (default 1)."""

        if seed is not None:
            np.random.seed(seed)

        if type(n) != int or n < 0 or p <= 0 or p >= 1:
            raise ValueError('n must be an integer >= 0. p must be in the range (0, 1).')

        return np.random.binomial(n, p, size=size)

    def negLogL(self, p, n, samples):
        """Calculate the negative log likelihood for a collection of random
        Bernoulli-distributed samples, and a specified p."""

        return (np.sum(scipy.special.comb(n, samples)) + np.sum(samples) * np.log(p) + (n * len(samples) - np.sum(samples)) * np.log(1 - p)) * -1

    def MLE(self, samples, n, use_minimizer=False, p0=None):
        """Returns the maximum likelihood estimate of parameter p, given a collection of samples.
        The Binomial parameter n must be known or estimated to use this function."""

        if use_minimizer:
            if p0 is None:
                raise ValueError('Supply an initial guess p0=p to the optimizer.')
            if p0 <= 0 or p0 >= 1:
                raise ValueError('p must be in the range (0, 1).')

            res = minimize(self.negLogL, p0, args=(n, samples), method='Nelder-Mead')
            if res.status == 1:
                print(f'Warning: Optimizer failed to converge with initial guess {p0}. Returned None for MLE values. Try another initial guess.')
                return None
            else:
                return res.x
        else:
            return np.array([np.mean(samples) / n])

    def GOF(self, samples, n, mle_p):
        """Returns the chi-squared goodness of fit statistic for a set of MLE paramters."""

        edges, f_exp = mergeBins(samples, scipy.stats.binom, n, mle_p)
        if edges is not None:
            f_obs, _ = np.histogram(a=samples, bins=edges+1)
            chisq, _ = scipy.stats.chisquare(f_obs=f_obs, f_exp=f_exp, ddof=len(f_obs)-2)

            return np.array([chisq, scipy.stats.chi2.isf(0.05, len(f_obs)-2)])
        else:
            return None, None

class Geometric(Display):

    def __init__(self):
        self.name = 'Geometric'
        self.measure_type = mt.DISCRETE
        self._parameters = [{'label': 'p', 'probability':[0,1]}]

    def sample(self, p, size=None, seed=None):
        """Get samples from Geom(p). The size argument is the number of samples (default 1)."""

        if seed is not None:
            np.random.seed(seed)

        if p <= 0 or p >= 1:
            raise ValueError('p must be in the range (0, 1).')

        return np.random.geometric(p=p, size=size)

    def negLogL(self, p, samples):
        """Returns the negative log likelihood given a collection of samples and parameter p."""

        n = len(samples)

        return (n * np.log(p) + np.sum(samples - 1) * np.log(1 - p)) * -1

    def MLE(self, samples, use_minimizer=False, p0=None):
        """Returns the maximum likelihood estimate of parameter p, given a collection of samples.
        If use_minimizer=True, an initial guess p0 for p must be provided. Otherwise, the closed
        form expression for the MLE of p is used."""

        if use_minimizer:
            if p0 is None:
                raise ValueError('Supply an initial guess p0=p to the optimizer.')
            if p0 <= 0 or p0 >= 1:
                raise ValueError('p must be in the range (0, 1). Supply an initial guess in this range.')
            res = minimize(self.negLogL, p0, args=samples, method='Nelder-Mead')
            if res.status == 1:
                print(f'Warning: Optimizer failed to converge with initial guess {p0}. Returned None for MLE values. Try another initial guess.')
                return None
            else:
                return res.x

        else:
            return np.array([1 / np.mean(samples)])

    def GOF(self, samples, mle_p):
        """Returns the chi-squared goodness of fit statistic for a set of MLE paramters."""

        chisq0, chisq = None, None

        edges, f_exp = mergeBins(samples, scipy.stats.geom, mle_p)
        if edges is not None:
            f_obs, _ = np.histogram(a=samples, bins=edges+1)
            ddof = len(f_obs) - 2
            if ddof > 0:
                chisq0, _ = scipy.stats.chisquare(f_obs=f_obs, f_exp=f_exp, ddof=ddof)
                chisq = scipy.stats.chi2.isf(0.05, ddof)

        return np.array([chisq0, chisq])

class Uniform(Display):

    def __init__(self):
        self.name = 'Uniform'
        self.measure_type = mt.CONTINUOUS
        self._parameters = [{'label':'a', 'range':[0,100], 'position':'min'}, {'label':'b', 'range':[0,100], 'position':'max'}]

    def sample(self, a=0., b=1., size=None, seed=None):
        """Get samples from Unif(a, b). The size argument is the number of samples (default 1)."""

        if seed is not None:
            np.random.seed(seed)

        return np.random.uniform(low=a, high=b, size=size)

    def MLE(self, samples):
        """Calculate the maximum likelihood estimator (MLE) for a collection of
        random uniformly distributed samples. Returns the MLE parameters a and b."""
        a = np.min(samples)
        b = np.max(samples)

        return np.array([a, b])

    def GOF(self, samples, mle_a, mle_b):
        """Returns the chi-squared goodness of fit statistic for a set of MLE paramters."""

        chisq0, chisq = None, None

        edges, f_exp = mergeBins(samples, scipy.stats.uniform, mle_a, mle_b)
        if edges is not None:
            f_obs, _ = np.histogram(a=samples, bins=edges)
            ddof = len(f_obs) - 3
            if ddof > 0:
                chisq0, _ = scipy.stats.chisquare(f_obs=f_obs, f_exp=f_exp, ddof=ddof)
                chisq = scipy.stats.chi2.isf(0.05, ddof)

        return np.array([chisq0, chisq])

class Normal(Display):

    def __init__(self):
        self.name = 'Normal'
        self.measure_type = mt.CONTINUOUS
        self._parameters = [{'label':'mean', 'span':[0,10]}, {'label':'var', 'span':[0,100]}]

    def sample(self, mean=0., var=1., size=None, seed=None):
        """Get samples from Norm(μ, σ^2). The size argument is the number of samples (default 1)."""

        if seed is not None:
            np.random.seed(seed)

        if var < 0:
            raise ValueError('var must be non-negative.')

        return np.random.normal(loc=mean, scale=np.sqrt(var), size=size)

    def negLogL(self, mean, var, samples):
        """Calculate the negative log likelihood for a collection of random
        normally distributed samples, and a specified mean and variance."""
        if var < 0:
            raise ValueError('var must be non-negative.')

        n = len(samples)

        return (-(n / 2) * np.log(2 * np.pi * var) - (1 / (2 * var)) * np.sum((samples - mean) ** 2)) * -1

    def MLE(self, samples, use_minimizer=False, mean0=None, var0=None):
        """Calculate the maximum likelihood estimator (MLE) for a collection of
        random normally distributed samples. Returns the MLE mean and variance.
        If use_minimizer=True, provide a initial guess for the optimizer in
        form mean0, var0."""

        if use_minimizer:
            if mean0 is None:
                raise ValueError('Supply an initial guess mean0 to the optimizer.')
            if var0 is None:
                raise ValueError('Supply an initial guess var0 to the optimizer.')
            if var0 < 0:
                raise ValueError('var0 must be non-negative. Supply an initial guess var0 with a positive var.')
            def nll(x, samples):
                return self.negLogL(*x, samples)
            res = minimize(nll, (mean0, var0), args=samples, method='Nelder-Mead')
            if res.status == 1:
                print(f'Warning: Optimizer failed to converge with initial guess {mean0}, {var0}. Returned None for MLE values. Try another initial guess.')
                return None, None
            else:
                return res.x

        else:
            mu = np.mean(samples)
            var = np.var(samples)

            return np.array([mu, var])

    def GOF(self, samples, mle_mu, mle_var):
        """Return the chi-squared goodness of fit statistic and p-value for a set of MLE paramters."""

        chisq0, chisq = None, None

        edges, f_exp = mergeBins(samples, scipy.stats.norm, mle_mu, np.sqrt(mle_var))
        if edges is not None:
            f_obs, _ = np.histogram(a=samples, bins=edges)
            ddof = len(f_obs) - 3
            if ddof > 0:
                chisq0, _ = scipy.stats.chisquare(f_obs=f_obs, f_exp=f_exp, ddof=ddof)
                chisq = scipy.stats.chi2.isf(0.05, ddof)

        return np.array([chisq0, chisq])

class Exponential(Display):

    def __init__(self):
        self.name = 'Exponential'
        self.measure_type = mt.CONTINUOUS
        self._parameters = [{'label':'lambd','span':[1,10]}]

    def sample(self, lambd=1., size=None, seed=None):
        """Get samples from Exp(λ). The size argument is the number of samples (default 1)."""

        if seed is not None:
            np.random.seed(seed)

        if not lambd > 0:
            raise ValueError('lambd must be greater than 0.')

        return np.random.exponential(scale=1/lambd, size=size)

    def negLogL(self, lambd, samples):
        """Calculate the negative log likelihood for a collection of random
        exponentially distributed samples, and a specified scale parameter lambd."""

        if not lambd > 0:
            raise ValueError('lambd must be greater than 0.')

        n = len(samples)

        return (n * np.log(lambd) - lambd * np.sum(samples)) * -1

    def MLE(self, samples, use_minimizer=False, lambd0=None):
        """Calculate the maximum likelihood estimator (MLE) for a collection of
        random normally distributed samples. Returns the MLE λ (lambd).
        If use_minimizer=True, provide a initial guess for the optimizer in
        form lambd0=lambd."""

        if use_minimizer:
            if lambd0 is None:
                raise ValueError('Supply an initial guess lambd0=lambd to the optimizer.')
            if lambd0 < 0:
                raise ValueError('lambd must be non-negative. Supply an positive initial guess lambd0=lambd.')
            res = minimize(self.negLogL, lambd0, args=samples)
            if res.status == 1:
                print(f'Warning: Optimizer failed to converge with initial guess {lambd0}. Returned None for MLE values. Try another initial guess.')
                return None
            else:
                return res.x

        else:
            lambd = 1 / np.mean(samples)
            return np.array([lambd])

    def GOF(self, samples, mle_lambda):
        """Return the chi-squared goodness of fit statistic and p-value for a set of MLE paramters."""

        chisq0, chisq = None, None

        edges, f_exp = mergeBins(samples, scipy.stats.expon, 1/mle_lambda)
        if edges is not None:
            f_obs, _ = np.histogram(a=samples, bins=edges)
            ddof = len(f_obs) - 2
            if ddof > 0:
                chisq0, _ = scipy.stats.chisquare(f_obs=f_obs, f_exp=f_exp, ddof=ddof)
                chisq = scipy.stats.chi2.isf(0.05, len(f_obs)-2)

        return np.array([chisq0, chisq])

class Gamma(Display):

    def __init__(self):
        self.name = 'Gamma'
        self.measure_type = mt.CONTINUOUS
        self._parameters = [{'label':'a', 'span':[0,100]}]

    def sample(self, a, b=1., size=None, seed=None):
        """Get samples from Gamma(a, b). The size argument is the number of samples (default 1)."""

        if seed is not None:
            np.random.seed(seed)

        if not a > 0 or not b > 0:
            raise ValueError('a and b must be greater than 0.')

        return np.random.gamma(shape=a, scale=b, size=size)

    def negLogL(self, a, b, samples):
        """Calculate the negative log likelihood for a collection of random
        gamma-distributed samples, and specified shape and scale parameters a and b."""

        n = len(samples)

        return ((a - 1) * np.sum(np.log(samples)) - n * np.log(scipy.special.gamma(a)) - n * a * np.log(b) - (np.sum(samples) / b)) * -1

    def MLE(self, samples, use_minimizer=False, a0=None, b0=None):
        """Calculate the maximum likelihood estimator (MLE) for a collection of
        random gamma-distributed samples. Returns the MLE a and b. Provide an
        initial guess for the optimizer in form a0, b0."""

        if use_minimizer:
            if a0 is None:
                raise ValueError('Supply an initial guess for a0 to the optimizer.')
            if b0 is None:
                raise ValueError('Supply an initial guess for b0 to the optimizer.')
            if not a0 > 0 or not b0 > 0:
                raise ValueError('a0 and b0 must be greater than 0. Supply an initial guess with a0 > 0 and b0 > 0.')

            def nll(x, samples):
                return self.negLogL(*x, samples)

            res = minimize(nll, (a0, b0), args=samples, method='Nelder-Mead')
            if res.status == 1:
                print(f'Warning: Optimizer failed to converge with initial guess {a0}, {b0}. Returned None for MLE values. Try another initial guess.')
                return None, None
            else:
                return res.x
        else:
            return gammaMLE(samples)

    def GOF(self, samples, mle_a, mle_b):
        """Return the chi-squared goodness of fit statistic and p-value for a set of MLE paramters."""

        chisq0, chisq = None, None

        edges, f_exp = mergeBins(samples, scipy.stats.gamma, mle_a, mle_b)
        if edges is not None:
            f_obs, _ = np.histogram(a=samples, bins=edges)
            ddof = len(f_obs) - 3
            if ddof > 0:
                chisq0, _ = scipy.stats.chisquare(f_obs=f_obs, f_exp=f_exp, ddof=ddof)
                chisq = scipy.stats.chi2.isf(0.05, ddof)

        return np.array([chisq0, chisq])

class Weibull(Display):

    def __init__(self):
        self.name = 'Weibull'
        self.measure_type = mt.CONTINUOUS
        self._parameters = [{'label':'a', 'span':[1,100]}]

    def sample(self, a, b=1, size=None, seed=None):
        """Get samples from Weibull(a, b). The shape parameter is a, the scale parameter is b (default 1).
        The size argument is the number of samples (default 1)."""

        if seed is not None:
            np.random.seed(seed)

        if not a > 0 or not b > 0:
            raise ValueError('a and b must be greater than 0.')

        return b * np.random.weibull(a=a, size=size)

    def negLogL(self, a, b, samples):
        """Calculate the negative log likelihood for a collection of random
        weibull-distributed samples, and specified shape and scale parameters a and b."""

        n = len(samples)

        return (n * np.log(a) - n * np.log(b) + (a - 1) * np.sum(np.log(samples / b)) - np.sum(np.power((samples / b), a))) * -1

    def MLE(self, samples, use_minimizer=False, a0=None, b0=None):
        """Calculate the maximum likelihood estimator (MLE) for a collection of
        random weibull-distributed samples. Returns the MLE a and b. Provide an
        initial guess for the optimizer in form a0, b0."""

        if use_minimizer:
            if a0 is None:
                raise ValueError('Supply an initial guess for a0 to the optimizer.')
            if b0 is None:
                raise ValueError('Supply an initial guess for b0 to the optimizer.')
            if not a0 > 0 or not b0 > 0:
                raise ValueError('a and b must be greater than 0. Supply an initial guess with a0 > 0 and b0 > 0.')

            def nll(x, samples):
                return self.negLogL(*x, samples)

            res = minimize(nll, (a0, b0), args=samples, method='Nelder-Mead')
            if res.status == 1:
                print(f'Warning: Optimizer failed to converge with initial guess {a0}, {b0}. Returned None for MLE values. Try another initial guess.')
                return None, None
            else:
                return res.x
        else:
            return weibullMLE(samples)

    def GOF(self, samples, mle_a, mle_b):
        """Return the chi-squared goodness of fit statistic and p-value for a set of MLE paramters."""

        chisq0, chisq = None, None

        edges, f_exp = mergeBins(samples, scipy.stats.weibull_min, mle_a, mle_b)
        if edges is not None:
            f_obs, _ = np.histogram(a=samples, bins=edges)
            ddof = len(f_obs) - 3
            if ddof > 0:
                chisq0, _ = scipy.stats.chisquare(f_obs=f_obs, f_exp=f_exp, ddof=ddof)
                chisq = scipy.stats.chi2.isf(0.05, ddof)

        return np.array([chisq0, chisq])

class Unknown(Display):

    def __init__(self):
        self.name = 'Unknown'
        self.measure_type = mt.UNKNOWN
