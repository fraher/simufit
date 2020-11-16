import numpy as np
import pandas as pd
from simufit.Dataset import Dataset
from scipy.optimize import minimize
from scipy.stats import norm, expon, gamma
import scipy.special
import sys
from simufit.Types import MeasureType as mt

import matplotlib
matplotlib.use('Qt5Agg')

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.colors as mcolors

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

        ### Menu and Toolbars ###
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        actionImport = QAction('Import data', self)
        actionImport.setShortcut(QtGui.QKeySequence('Ctrl+I'))
        actionImport.setStatusTip('Import a data file')
        actionImport.triggered.connect(self.importData)
        fileMenu.addAction(actionImport)

        self.sc = MplCanvas(self, width=8, height=6, dpi=100)
        self.sc.setSizePolicy(qsp.Fixed, qsp.Fixed)

        # Create toolbar, passing canvas as first parameter, parent (self, the MainWindow) as second.
        toolbar = NavigationToolbar(self.sc, self)

        # Create grid layout for selecting distribution
        self.distSelector = QtWidgets.QComboBox()
        self.distSelector.setFixedSize(QtCore.QSize(100, 24))
        if self.dist is None:
            dists = ['Geometric', 'Uniform', 'Normal', 'Exponential', 'Gamma']
            for dist in dists:
                self.distSelector.addItem(dist)
        else:
            self.distSelector.addItem(self.dist.name)
        self.distSelector.currentTextChanged.connect(self.changeDist)

        # Slider 1
        self.slider1Label = QtWidgets.QLabel('p')
        self.slider1 = QtWidgets.QSlider(minimum=1, orientation=QtCore.Qt.Horizontal, maximum=999)
        self.slider1Value = QtWidgets.QLabel('0.01')
        self.slider1.valueChanged[int].connect(self.plotData)
        self.slider1.setFixedWidth(275)

        # Slider 2
        self.slider2Label = QtWidgets.QLabel('Mean')
        self.slider2 = QtWidgets.QSlider(minimum=-999, orientation=QtCore.Qt.Horizontal, maximum=999)
        self.slider2.setValue(0)
        self.slider2Value = QtWidgets.QLabel('0')
        self.slider2.valueChanged[int].connect(self.plotData)
        self.slider2.setFixedWidth(275)

        # Slider 3
        self.slider3Label = QtWidgets.QLabel('Variance')
        self.slider3 = QtWidgets.QSlider(minimum=1, orientation=QtCore.Qt.Horizontal, maximum=999)
        self.slider3Value = QtWidgets.QLabel('0.01')
        self.slider3.valueChanged[int].connect(self.plotData)
        self.slider3.setFixedWidth(275)

        # Slider 4
        # TODO: Set number of bins here

        dist = QtWidgets.QGridLayout()
        dist.addWidget(QtWidgets.QLabel('Distribution:'), 0, 0)
        dist.addWidget(self.distSelector, 0, 1)
        dist.addItem(QtWidgets.QSpacerItem(0, 0, qsp.Expanding, qsp.Fixed), 0, 2)

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

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(toolbar)
        layout.addLayout(dist)
        layout.addItem(QtWidgets.QSpacerItem(0, 15, qsp.Expanding, qsp.Fixed))
        layout.addLayout(self.sliders1)
        layout.addLayout(self.sliders2)
        layout.addItem(QtWidgets.QSpacerItem(0, 15, qsp.Expanding, qsp.Fixed))
        layout.addLayout(self.sliders3)
        layout.addWidget(self.sc)

        # Create a placeholder widget to hold our toolbar and canvas.
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def importData(self):
        """Import a data file (csv, txt) and view as histogram."""

        self.datasetWindow.show()

    def changeDist(self):

        if self.dist is not None:
            dist = self.dist.name
        else:
            dist = self.distSelector.currentText()

        slider1Widgets = (self.sliders1.itemAt(i).widget() for i in range(self.sliders1.count()) if not isinstance(self.sliders1.itemAt(i), QtWidgets.QSpacerItem))
        slider2Widgets = (self.sliders2.itemAt(i).widget() for i in range(self.sliders2.count()) if not isinstance(self.sliders2.itemAt(i), QtWidgets.QSpacerItem))
        slider3Widgets = (self.sliders3.itemAt(i).widget() for i in range(self.sliders3.count()) if not isinstance(self.sliders3.itemAt(i), QtWidgets.QSpacerItem))

        if dist in ['Geometric', 'Exponential']:
            for w in slider1Widgets:
                w.show()
            for w in slider2Widgets:
                w.hide()
            for w in slider3Widgets:
                w.hide()
            if dist == 'Geometric':
                self.slider1Label.setText('p')
            elif dist == 'Exponential':
                self.slider1Label.setText('λ')

        if dist in ['Normal', 'Gamma', 'Weibull']:
            for w in slider1Widgets:
                w.hide()
            for w in slider2Widgets:
                w.show()
            for w in slider3Widgets:
                w.show()
            if dist == 'Normal':
                self.slider2.setMinimum(-999)
                self.slider3.setMinimum(-999)
                self.slider2Label.setText('Mean')
                self.slider3Label.setText('Variance')
            elif dist == 'Gamma' or dist == 'Weibull':
                self.slider2.setMinimum(1)
                self.slider3.setMinimum(1)
                self.slider2Label.setText('a')
                self.slider3Label.setText('b')

        QtCore.QTimer.singleShot(10, lambda: self.resize(self.minimumSize()))

    def plotData(self):

        dist = self.distSelector.currentText()
        self.sc.axes.clear()

        if dist == 'Geometric':
            self.sc.axes.hist(self.samples, bins=np.max(self.samples), density=True, color=(152/255, 200/255, 132/255), ec='white')
        else:
            self.sc.axes.hist(self.samples, bins=np.histogram_bin_edges(self.samples, 'fd'), density=True, color=(152/255, 200/255, 132/255), ec='white')

        x = ""
        f = ""

        if dist == 'Geometric':
            x = np.arange(1, np.max(self.samples))
            p = self.slider1.value() / 1000
            f = ((1 - p) ** (x - 1)) * p
            self.slider1Value.setText(str(round(p, 3)))
        elif dist == 'Exponential':
            lambd = self.slider1.value() / 100
            x = np.linspace(expon.ppf(0.001, scale=1/lambd), expon.ppf(0.999, scale=1/lambd), 100)
            f = expon.pdf(x, scale=1/lambd)
            self.slider1Value.setText(str(round(lambd, 3)))
        elif dist == 'Normal':
            mean = self.slider2.value() / 100
            var = self.slider3.value() / 100
            std = np.sqrt(var)
            self.slider2Value.setText(str(round(mean, 3)))
            self.slider3Value.setText(str(round(var, 3)))
            x = np.linspace(norm.ppf(0.001, loc=mean, scale=std), norm.ppf(0.999, loc=mean, scale=std), 100)
            f = norm.pdf(x, loc=mean, scale=std)
        elif dist == 'Gamma':
            a = self.slider2.value() / 100
            b = self.slider3.value() / 100
            self.slider2Value.setText(str(round(a, 3)))
            self.slider3Value.setText(str(round(b, 3)))
            x = np.linspace(gamma.ppf(0.001, a, scale=b), gamma.ppf(0.999, a, scale=b), 100)
            f = gamma.pdf(x, a, scale=b)

        self.sc.axes.plot(x, f)
        self.sc.fig.canvas.draw_idle()

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
        if comparison_distribution.getSamples() is not None:
            if len(samples) != len(comparison_distribution.getSamples()):
                print("Distribution sample sizes do not match")
                return
        show_histogram(samples, bins, comparison_distribution)

class Bernoulli(Display):

    def __init__(self):
        self.name = 'Bernoulli'
        self.measure_type = mt.DISCRETE

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

        return samples

    def negLogL(self, p, samples):
        """Calculate the negative log likelihood for a collection of random
        Bernoulli-distributed samples, and a specified p."""

        n = len(samples)
        m = np.sum(samples)

        return (m * np.log(p) + (n - m) * np.log(1 - p)) * -1

    def MLE(self, samples, use_minimizer=False, x0=None):
        """Returns the maximum likelihood estimate of parameter p, given a collection of samples.
        If use_minimizer=True, an initial guess x0 for p must be provided. Otherwise, the closed
        form expression for the MLE of p is used."""

        if use_minimizer:
            if x0 is None:
                raise ValueError('Supply an initial guess x0=p to the optimizer.')
            if x0 <= 0 or x0 >= 1:
                raise ValueError('p must be in the range (0, 1). Supply an initial guess in this range.')
            return minimize(self.negLogL, x0, args=samples, method='Nelder-Mead')

        else:
            return np.mean(samples)

class Geometric(Display):

    def __init__(self):
        self.name = 'Geometric'
        self.measure_type = mt.DISCRETE

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

    def MLE(self, samples, use_minimizer=False, x0=None):
        """Returns the maximum likelihood estimate of parameter p, given a collection of samples.
        If use_minimizer=True, an initial guess x0 for p must be provided. Otherwise, the closed
        form expression for the MLE of p is used."""

        if use_minimizer:
            if x0 is None:
                raise ValueError('Supply an initial guess x0=p to the optimizer.')
            if x0 <= 0 or x0 >= 1:
                raise ValueError('p must be in the range (0, 1). Supply an initial guess in this range.')
            return minimize(self.negLogL, x0, args=samples, method='Nelder-Mead')

        else:
            return 1 / np.mean(samples)

class Uniform(Display):

    def __init__(self):
        self.name = 'Uniform'
        self.measure_type = mt.CONTINUOUS

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

        return a, b

class Normal(Display):

    def __init__(self):
        self.name = 'Normal'
        self.measure_type = mt.CONTINUOUS

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

    def MLE(self, samples, use_minimizer=False, x0=None):
        """Calculate the maximum likelihood estimator (MLE) for a collection of
        random normally distributed samples. Returns the MLE mean and variance.
        If use_minimizer=True, provide a initial guess for the optimizer in
        form x0=(mean_guess, var_guess)."""

        if use_minimizer:
            if x0 is None:
                raise ValueError('Supply an initial guess x0=(mean, var) to the optimizer.')
            if x0[1] < 0:
                raise ValueError('var must be non-negative. Supply an initial guess x0=(mean, var) with a positive var.')
            def nll(x, samples):
                return self.negLogL(*x, samples)
            return minimize(nll, x0, args=samples, method='Nelder-Mead')

        else:
            mu = np.mean(samples)
            var = np.std(samples) ** 2

            return mu, var

class Exponential(Display):

    def __init__(self):
        self.name = 'Exponential'
        self.measure_type = mt.CONTINUOUS

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

    def MLE(self, samples, use_minimizer=False, x0=None):
        """Calculate the maximum likelihood estimator (MLE) for a collection of
        random normally distributed samples. Returns the MLE λ (lambd).
        If use_minimizer=True, provide a initial guess for the optimizer in
        form x0=lambd."""

        if use_minimizer:
            if x0 is None:
                raise ValueError('Supply an initial guess x0=lambd to the optimizer.')
            if x0 < 0:
                raise ValueError('lambd must be non-negative. Supply an positive initial guess x0=lambd.')
            return minimize(self.negLogL, x0, args=samples)

        else:
            n = len(samples)
            lambd = n / np.sum(samples)
            return lambd

class Gamma(Display):

    def __init__(self):
        self.name = 'Gamma'
        self.measure_type = mt.CONTINUOUS

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

        return ((a - 1) * np.sum(np.log(samples)) - n * scipy.special.gamma(a) - n * a * np.log(b) - (np.sum(samples) / b)) * -1

    def MLE(self, samples, x0=None):
        """Calculate the maximum likelihood estimator (MLE) for a collection of
        random gamma-distributed samples. Returns the MLE a and b.
        If use_minimizer=True, provide a initial guess for the optimizer in
        form x0=(a, b)."""

        if x0 is None:
            raise ValueError('Supply an initial guess x0=(a,b) to the optimizer.')
        if not x0[0] > 0 or not x0[1] > 0:
            raise ValueError('a and b must be greater than 0. Supply an initial guess x0=(a,b) with a > 0 and b > 0.')

        def nll(x, samples):
            return self.negLogL(*x, samples)
        return minimize(nll, x0, args=samples, method='Nelder-Mead')

class Weibull(Display):

    def __init__(self):
        self.name = 'Weibull'
        self.measure_type = mt.CONTINUOUS

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

        return (n * np.log(b) - n * np.log(a) + (b - 1) * np.sum(np.log(samples / a)) - np.sum(np.power((samples / a), b))) * -1

    def MLE(self, samples, x0=None):
        """Calculate the maximum likelihood estimator (MLE) for a collection of
        random weibull-distributed samples. Returns the MLE a and b.
        If use_minimizer=True, provide a initial guess for the optimizer in
        form x0=(a, b)."""

        if x0 is None:
            raise ValueError('Supply an initial guess x0=(a,b) to the optimizer.')
        if not x0[0] > 0 or not x0[1] > 0:
            raise ValueError('a and b must be greater than 0. Supply an initial guess x0=(a,b) with a > 0 and b > 0.')

        def nll(x, samples):
            return self.negLogL(*x, samples)
        return minimize(nll, x0, args=samples, method='Nelder-Mead')