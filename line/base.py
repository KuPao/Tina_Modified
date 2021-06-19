from ..common import *


@ti.data_oriented
class LineEditBase:
    def __init__(self, line):
        self.line = line

    def __getattr__(self, attr):
        return getattr(self.line, attr)
