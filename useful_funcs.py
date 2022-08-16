class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as mtick

"""### Plots"""

def plt_config(title=None, xlim=None, ylim=None, xlabel=None, ylabel=None, colorbar=False, sci=False):
    for field in ['title', 'xlim', 'ylim', 'xlabel', 'ylabel']:
        if eval(field) != None: getattr(plt, field)(eval(field))
    if isinstance(sci, str): plt.ticklabel_format(style='sci', axis=sci, scilimits=(0,0))
    if isinstance(colorbar,str): plt.colorbar(label=colorbar)
    elif colorbar: plt.colorbar(label = '$Number\ of\ Entries$')

@contextmanager
def initiate_plot(dimx=24, dimy=9, dpi=100, fontweight='normal'):
    plt.rcParams['figure.figsize'] = (dimx, dimy)
    plt.rcParams['font.weight'] = fontweight
    plt.rcParams['mathtext.default'] = 'regular'
    plt.rcParams["font.family"] = 'Arial'
    global fig; fig = plt.figure(dpi=dpi)
    yield
    plt.show()
