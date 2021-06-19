from ..common import *


@ti.data_oriented
class SimpleLine:
    def __init__(self, radius=0.02):
        self.verts = ti.Vector.field(3, float, 2)
        self.sizes = ti.field(float, 2)
        self.colors = ti.Vector.field(3, float, 2)
        self.nline = ti.field(int, ())

        @ti.materialize_callback
        def init_pars():
            self.sizes.fill(radius)
            self.colors.fill(1)

    @ti.func
    def pre_compute(self):
        pass

    @ti.func
    def get_nlines(self):
        return min(self.nline[None], 2)

    @ti.func
    def get_line_verts(self):
        return [self.verts[0], self.verts[1]]

    @ti.func
    def get_particle_position(self, n):
        return self.verts[n]

    @ti.func
    def get_particle_radius(self, n):
        return self.sizes[n]

    @ti.func
    def get_line_color(self, n):
        return self.colors[n]

    @ti.kernel
    def set_lines(self, verts: ti.ext_arr()):
        self.nline[None] = min(verts.shape[0], self.verts.shape[0])
        for i in range(self.nline[None]):
            for k in ti.static(range(3)):
                self.verts[i][k] = verts[i, k]

    @ti.kernel
    def set_particle_radii(self, sizes: ti.ext_arr()):
        for i in range(self.nline[None]):
            self.sizes[i] = sizes[i]

    @ti.kernel
    def set_line_colors(self, colors: ti.ext_arr()):
        for i in range(self.nline[None]):
            for k in ti.static(range(3)):
                self.colors[i][k] = colors[i, k]
