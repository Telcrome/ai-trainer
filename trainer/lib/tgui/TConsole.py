from qtconsole.inprocess import QtInProcessKernelManager
from qtconsole.rich_jupyter_widget import RichJupyterWidget


class TConsole(RichJupyterWidget):

    def __init__(self):
        super().__init__()
        kernel_manager = QtInProcessKernelManager()
        kernel_manager.start_kernel()
        self.kernel = kernel_manager.kernel
        self.kernel.gui = 'qt4'
        self.kernel.shell.push({'console_object': self})

        # kernel_client = kernel_manager.client()
        # kernel_client.start_channels()
        self.kernel_manager = kernel_manager
        # self.kernel_client = kernel_client

        def stop():
            print("This does not work yet")

        self.exit_requested.connect(stop)

    def push_to_ipython(self, x):
        self.kernel.shell.push(x)
