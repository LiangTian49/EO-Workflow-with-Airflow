import logging
import psutil

class CustomLogger(logging.Logger):

    def __init__(self, name):
        super(CustomLogger, self).__init__(name)

    def info(self, msg, *args, **kwargs):
        used_memory = psutil.virtual_memory().used / pow(2, 30)
        total_memory = psutil.virtual_memory().total / pow(2, 30)
        msg = "[Mem (GiB): {0:.2f}/{1:.2f} --- CPU (%): {2}] ".format(used_memory, total_memory, psutil.cpu_percent()) + msg
        super().info(msg, *args, **kwargs)
