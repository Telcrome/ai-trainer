import numpy as np
from flexx import flx
from bokeh.plotting import figure

# import trainer.lib as lib

x = np.linspace(0, 6, 50)
p1 = figure()
p1.line(x, np.sin(x))

p2 = figure()
p2.line(x, np.cos(x))


class DebuggerGui(flx.PyWidget):

    # def __init__(self):
    #     # self.bs: List[Optional[flx.Button]] = [None, None, None]
    #     super().__init__()

    def init(self):
        with flx.HBox():
            with flx.VBox():
                self.bs0 = flx.Button(text='Button1', flex=0)
                self.bs1 = flx.Button(text='Button2', flex=1)
                self.bs2 = flx.Button(text='Button3', flex=2)
                self.prog = flx.ProgressBar(flex=1, value=0.1, text='{percent} done')
                self.lbl_placeholder = flx.Label(flex=1, style='overflow-y: scroll;')
            with flx.VBox():
                self.lbl = flx.Label(flex=1, style='overflow-y: scroll;')
                # flx.BokehWidget.from_plot(p1)
                # flx.BokehWidget.from_plot(p2)


if __name__ == '__main__':
    logger = flx.App(DebuggerGui)
    logger.export('logger.html', link=0)

    if __name__ == '__main__':
        logger.launch('browser')
        flx.run()
