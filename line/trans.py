from ..common import *
from .base import LineEditBase


class LineTransform(LineEditBase):
    def __init__(self, line):
        super().__init__(line)

        self.trans = ti.Matrix.field(4, 4, float, ())
        self.scale = ti.field(float, ())

        @ti.materialize_callback
        @ti.kernel
        def init_trans():
            self.trans[None] = ti.Matrix.identity(float, 4)
            self.scale[None] = 1

    def set_transform(self, trans, scale):
        self.trans[None] = np.array(trans).tolist()
        self.scale[None] = scale

    @ti.func
    def get_particle_position(self, n):
        vert = self.line.get_particle_position(n)
        return mapply_pos(self.trans[None], vert)
