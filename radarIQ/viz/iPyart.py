import pyart
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from pathlib import Path
from . import cmaps

def getPlottableFields(nyquistVel: float, negDealias: int = 0, posDealias: int = 0): 
    fields = {
        'DBZ': {
            'cmap': 'pyart_Carbone42',
            'title': 'Equivalent Reflectivity Factor',
            'norm': Normalize(vmin=0, vmax=75)
        },
        'VEL': {
            'cmap': 'pyart_Carbone42',
            'title': 'Radial Velocity',
            'norm': TwoSlopeNorm(vmin=-nyquistVel - 2*negDealias*nyquistVel, vcenter=0, vmax=nyquistVel + 2*posDealias*nyquistVel)
        },
        'ZDR': {
            'cmap': cmaps.dmap(256),
            'title': 'Differential Reflectivity',
            'norm': Normalize(vmin=-5, vmax=8)
        },
        'RHOHV': {
            'cmap': cmaps.rmap(256),
            'title': 'Correlation Coefficient',
            'norm': Normalize(vmin=0.1, vmax=1.05)
        } 
    }

    return fields

class iPyart:
    def __init__(self, cfradDir, startFilenum):
        if matplotlib.get_backend() != 'TkAgg':
            raise RuntimeError("Cannot run interactive display with current backend. "
                               "Switch to Tkagg with 'matplotlib.use(backend=\"TkAgg\")' "
                               "or if using jupyter, '%matplotlib tk.'")
        self.files = list(Path(cfradDir).iterdir())
        self.maxFiles = len(self.files)
        self.curFileNum = startFilenum
        self.curFileName = self.files[self.curFileNum].name
        self.currentRadarObj = pyart.io.read(self.files[self.curFileNum])
        self.singleFileMode = False

        self.customPlots = []
        self.handlers = []

        self.isShowing = False

    def _replotAllCustomPlots(self):
        for arguments in self.customPlots:
            self.plot(arguments, True)

    def _timeSwitchManager(self, event):
        if self.singleFileMode:
            return
        allowedKeys = ['left', 'right']
        if not (event.key in allowedKeys):
            return
        
        if event.key == 'left':
            self._replotDiffScan(-1)
        elif event.key == 'right':
            self._replotDiffScan(1)
        
        self._replotAllCustomPlots()
        event.canvas.draw()
        event.canvas.flush_events()

    def _replotDiffScan(self, direction: int):
        if not self.isShowing:
            raise RuntimeError("Need to have a plot showing.")
        if not (direction == -1 or direction == 1):
            raise ValueError("Invalid direction argument. Only -1 or 1 valid.")
        if not (((self.curFileNum+direction) < self.maxFiles) or ((self.curFileNum+direction) >= 0)):
            return
        
        plt.figure(self.identifier)
        current_xlims = self.radarDisplay.plots[0].axes.get_xlim()
        current_ylims = self.radarDisplay.plots[0].axes.get_ylim()
        plt.clf()
        
        self.curFileNum += direction
        self.curFileName = self.files[self.curFileNum].name
        self.currentRadarObj = pyart.io.read(self.files[self.curFileNum])
        fieldPlotting = getPlottableFields(self.currentRadarObj.get_nyquist_vel(0))[self.currentField]
        norm = fieldPlotting['norm']
        cmap = fieldPlotting['cmap']

        self.radarDisplay = pyart.graph.RadarDisplay(self.currentRadarObj)
        self.radarDisplay.set_limits(xlim=current_xlims, ylim=current_ylims)
        self.radarDisplay.plot_ppi(self.currentField, norm=norm, cmap=cmap)

    def _fieldSwitchManager(self, event):
        allowedKeys = ['z', 'v', 'd', 'r']
        if not (event.key in allowedKeys):
            return

        if event.key == 'z':
            self._replotSameScan('DBZ')
        elif event.key == 'v':
            self._replotSameScan('VEL')
        elif event.key == 'd':
            self._replotSameScan('ZDR')
        elif event.key == 'r':
            self._replotSameScan('RHOHV')

        self._replotAllCustomPlots()
        event.canvas.draw()
        event.canvas.flush_events()

    def _replotSameScan(self, field: str):
        if not self.isShowing:
            raise RuntimeError("Need to have a plot showing.")
        if self.currentField == field:
            return

        fields = getPlottableFields(self.currentRadarObj.get_nyquist_vel(0))
        if not (field in fields.keys()):
            raise ValueError("Requested field is not plottable.")
        
        plt.figure(self.identifier)
        current_xlims = self.radarDisplay.plots[0].axes.get_xlim()
        current_ylims = self.radarDisplay.plots[0].axes.get_ylim()
        plt.clf()
        self.currentField = field
        fieldPlotting = getPlottableFields(self.currentRadarObj.get_nyquist_vel(0))[field]
        norm = fieldPlotting['norm']
        cmap = fieldPlotting['cmap']

        self.radarDisplay = pyart.graph.RadarDisplay(self.currentRadarObj)
        self.radarDisplay.set_limits(xlim=current_xlims, ylim=current_ylims)
        self.radarDisplay.plot_ppi(field, norm=norm, cmap=cmap)
        

    def open(self):
        if self.isShowing:
            raise RuntimeError("Plot is already showing.")

        self.fig = plt.figure(figsize = [10,8])
        self.identifier = self.fig.number
        self.currentField = 'DBZ'

        initField = getPlottableFields(self.currentRadarObj.get_nyquist_vel(0))['DBZ']
        norm = initField['norm']
        cmap = initField['cmap']
        self.radarDisplay = pyart.graph.RadarDisplay(self.currentRadarObj)
        self.radarDisplay.plot_ppi('DBZ', norm=norm, cmap=cmap)
        self.isShowing = True

        manager = plt.get_current_fig_manager()
        if hasattr(manager, 'window'):
            root = manager.window
            root.protocol("WM_DELETE_WINDOW", self.close)

        self.fig.canvas.mpl_connect('key_press_event', self._fieldSwitchManager)
        self.fig.canvas.mpl_connect('key_press_event', self._timeSwitchManager)

        for action, handler in self.handlers:
            self.fig.canvas.mpl_connect(action, handler)

        plt.show(block=True)


    def close(self):
        if not self.isShowing:
            raise RuntimeError("Plot must be showing before it can be closed.")
        
        plt.figure(self.identifier)
        plt.close()
        self.isShowing = False
        self.fig = None
        self.identifier = None
        self.currentField = None
        self.radarDisplay = None


    def plot(self, arguments, replotting: bool = False):
        if not self.isShowing:
            raise RuntimeError("Need to have a plot showing.")
        plt.figure(self.identifier)

        if not (isinstance(arguments, tuple) and len(arguments) == 4):
            raise ValueError("Argument must be a tuple with four elements.")
        plt_type, plt_x, plt_y, plt_color = arguments
        if not isinstance(plt_type, str):
            raise TypeError("First element must be a string.")
        if not isinstance(plt_x, (int, float, list, np.ndarray)):
            raise TypeError("Second element must be a number or an array.")
        if not isinstance(plt_y, (int, float, list, np.ndarray)):
            raise TypeError("Third element must be a number or an array.")
        if not isinstance(plt_color, str):
            raise TypeError("Fourth element must be a string.")
        allowedTypes = ['scatter', 'plot']
        if not (plt_type in allowedTypes):
            raise ValueError("Only 'scatter' and 'plot' supported for first argument.")

        if plt_type == 'scatter':
            plt.scatter(plt_x, plt_y, color=plt_color)
        elif plt_type == 'plot':
            plt.plot(plt_x, plt_y, color=plt_color)

        if not replotting:
            self.customPlots.append(arguments)


    def getFileNum(self):
        return self.curFileNum
    
    
    def addCustomHandler(self, action: str, handler):
        self.handlers.append((action, handler))