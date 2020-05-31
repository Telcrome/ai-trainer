import os
from flexx import flx


class Example(flx.PyWidget):

    def __init__(self):
        self.button, self.lbl = None, None
        super().__init__()

    def init(self):
        self.button = flx.Button(text='hello')
        self.lbl = flx.Label(flex=1, style='overflow-y: scroll;')

    @flx.reaction('button.pointer_click')
    def some_stuff(self, *events):
        self.lbl.set_html('<br />'.join(os.listdir('.')))
        print(events)


app = flx.App(Example)
app.export('example.html', link=0)

app.launch('browser')
flx.run()
