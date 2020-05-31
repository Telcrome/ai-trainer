import os
from flexx import app, ui, flx

class Example(flx.PyWidget):

    def init(self):
        self.button = flx.Button(text='hello')
        self.lbl = flx.Label(flex=1, style='overflow-y: scroll;')

    @flx.reaction('button.pointer_click')
    def some_stuff(self, *events):
        # with ui.TreeWidget(max_selected=2):

        #     for t in ['foo', 'bar', 'spam', 'eggs']:
        #         ui.TreeItem(text=t, checked=True)
        # self.lbl.set_html('1<br/>2')
        self.lbl.set_html('<br />'.join(os.listdir('.')))

app = flx.App(Example)
app.export('example.html', link=0)

app.launch('browser')
flx.run()
