import numpy as np
from scipy.optimize import minimize
import scipy.special
import sys

import matplotlib
matplotlib.use('Qt5Agg')

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from PyQt5 import QtWidgets, QtGui, QtCore, QtSql
from PyQt5.QtWidgets import QWidget, QApplication, QMainWindow, QAction, QFileDialog, QInputDialog
from PyQt5.QtGui import QPalette, QWheelEvent, QCursor, QPixmap


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_facecolor((53/255, 53/255, 53/255))
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)


class MainWindow(QMainWindow):

    def __init__(self, samples):
        super(MainWindow, self).__init__()
        self.setWindowTitle('Distribution Fitter')
        self.samples = samples
        self.initUI()

    def initUI(self):
        """Sets up all the UI functionality."""

        ### Menu and Toolbars ###

        self.sc = MplCanvas(self, width=8, height=6, dpi=100)
        self.sc.axes.set_facecolor((53/255, 53/255, 53/255))
        self.sc.axes.hist(self.samples, bins=np.arange(10), density=True, color=(152/255, 200/255, 132/255), ec='white')

        # Create toolbar, passing canvas as first parameter, parent (self, the MainWindow) as second.
        toolbar = NavigationToolbar(self.sc, self)

        # Create grid layout for selecting distribution
        self.dist_selector = QtWidgets.QComboBox()
        for item in ['Geometric', 'Uniform', 'Normal', 'Exponential', 'Gamma']:
            self.dist_selector.addItem(item)

        self.pLabel = QtWidgets.QLabel('p')
        self.pSlider = QtWidgets.QSlider(minimum=1, orientation=QtCore.Qt.Horizontal, maximum=99)
        self.pValue = QtWidgets.QLabel('0.01')
        self.pSlider.valueChanged[int].connect(self.sliderEvent)
        self.pSlider.setFixedWidth(275)

        grid = QtWidgets.QGridLayout()
        grid.addWidget(self.dist_selector, 0, 0)
        grid.addWidget(self.pLabel, 0, 1)
        grid.addWidget(self.pValue, 0, 2)
        grid.addWidget(self.pSlider, 0, 3)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(toolbar)
        layout.addLayout(grid)
        layout.addWidget(self.sc)

        # Create a placeholder widget to hold our toolbar and canvas.
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def sliderEvent(self):

        if len(self.sc.axes.lines) > 0:
            self.sc.axes.lines.pop()

        k = np.arange(1, 10)
        p = self.pSlider.value() / 100
        f = ((1 - p) ** (k - 1)) * p
        self.sc.axes.plot(k, f)
        self.sc.fig.canvas.draw_idle()

        self.pValue.setText(str(round(p, 3)))


def window(samples):
    if not QApplication.instance():
        app = QApplication(sys.argv)
    else:
        app = QApplication.instance()

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

    win = MainWindow(samples)
    win.show()
    QtWidgets.QApplication.setQuitOnLastWindowClosed(True)
    app.exec_()
    app.quit()


class Geometric:

    def __init__(self):
        pass

    def sample(self, p, size=None):
        """Get samples from Geom(p). The size argument is the number of samples (default 1)."""

        if p <= 0 or p >= 1:
            raise ValueError('p must be in the range (0, 1).')

        return np.random.geometric(p=p, size=size)

    def negLogL(self, p, samples):

        n = len(samples)

        return (n * np.log(p) + np.sum(samples - 1) * np.log(1 - p)) * -1

    def MLE(self, samples, use_minimizer=False, x0=None):

        if use_minimizer:
            if x0 is None:
                raise ValueError('Supply an initial guess x0=p to the optimizer.')
            if x0 <= 0 or x0 >= 1:
                raise ValueError('p must be in the range (0, 1). Supply an initial guess in this range.')
            return minimize(self.negLogL, x0, args=samples, method='Nelder-Mead')

        else:
            return 1 / np.mean(samples)

    def fit(self, samples):
        window(samples)


class Uniform:

    def __init__(self):
        pass

    def sample(self, a=0., b=1., size=None):
        """Get samples from Unif(a, b). The size argument is the number of samples (default 1)."""

        return np.random.uniform(low=a, high=b, size=size)

    def MLE(self, samples):
        """Calculate the maximum likelihood estimator (MLE) for a collection of
        random uniformly distributed samples. Returns the MLE parameters a and b."""

        a = np.min(samples)
        b = np.max(samples)

        return a, b

    def fit(self, samples):
        window(samples)


class Normal:

    def __init__(self):
        pass

    def seed(self, seed):
        """Set seed."""

        np.random.seed(seed)

    def sample(self, mean=0., var=1., size=None):
        """Get samples from Norm(μ, σ^2). The size argument is the number of samples (default 1)."""

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

    def fit(self, samples):
        window(samples)

class Exponential:

    def init(self):
        pass

    def sample(self, lambd=1., size=None):
        """Get samples from Exp(λ). The size argument is the number of samples (default 1)."""

        if lambd < 0:
            raise ValueError('lambd must be non-negative.')

        return np.random.exponential(scale=1/lambd, size=size)

    def negLogL(self, lambd, samples):
        """Calculate the negative log likelihood for a collection of random
        exponentially distributed samples, and a specified scale parameter lambd."""

        if lambd < 0:
            raise ValueError('lambd must be non-negative.')

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

    def fit(self, samples):
        window(samples)

class Gamma:

    def __init__(self):
        pass

    def sample(self, a, b=1., size=None):
        """Get samples from Gamma(a, b). The size argument is the number of samples (default 1)."""

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

    def fit(self, samples):
        window(samples)

