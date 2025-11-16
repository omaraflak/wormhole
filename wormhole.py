import numpy as np
import taichi as ti

ti.init(arch=ti.gpu, default_fp=ti.f32)

Vector6f32 = ti.types.vector(6, ti.f32)
Vector3f32 = ti.types.vector(3, ti.f32)
Field = ti.template()


@ti.func
def r_of_l(l: float, r0: float) -> float:
    return ti.sqrt(l * l + r0 * r0)


@ti.func
def r_prime_of_l(l: float, r0: float) -> float:
    return l / ti.sqrt(l * l + r0 * r0)


@ti.func
def rhs(state: Vector6f32, r0: float) -> Vector6f32:
    l, th, ph, vl, vth, vph = state[0], state[1], state[2], state[3], state[4], state[5]

    r = r_of_l(l, r0)
    rp = r_prime_of_l(l, r0)
    rpr = rp / r
    st = ti.sin(th)
    ct = ti.cos(th)

    # make sure we don't divide by zero by adding an epsilon
    # also make sure we don't flip the sign by making epsilon signed
    epsilon = 1e-12
    st_safe = ti.select(ti.abs(st) < epsilon, ti.math.sign(st) * epsilon, st)
    st_safe = ti.select(st == 0.0, epsilon, st_safe)

    # geodesic equations in wormhole space
    al = r * rp * (vth * vth + st * st * vph * vph)
    ath = -2.0 * rpr * vth * vl + st * ct * vph * vph
    aph = -2.0 * rpr * vph * vl - 2 * (ct / st_safe) * vph * vth

    return ti.Vector([vl, vth, vph, al, ath, aph], dt=ti.f32)


@ti.func
def rk4_step(state: Vector6f32, h: float, r0: float) -> Vector6f32:
    k1 = rhs(state, r0)
    k2 = rhs(state + 0.5 * h * k1, r0)
    k3 = rhs(state + 0.5 * h * k2, r0)
    k4 = rhs(state + h * k3, r0)
    return state + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


@ti.func
def trace_geodesic(state: Vector6f32, h: float, hmax: float, r0: float) -> Vector3f32:
    ch = 0.0
    while ch < hmax:
        ch += h
        state = rk4_step(state, h, r0)
        if ti.abs(state[0]) > 6:
            break
    # return the last (l, theta, phi) of the trace
    return ti.Vector([state[0], state[1], state[2]], dt=ti.f32)


@ti.kernel
def make_initial_states_kernel(l0: float, th0: float, ph0: float, r0: float, fov: float, height: int, width: int, states: Field):
    half_h = ti.tan(fov / 2.0)
    half_w = half_h * (width / height)
    r = r_of_l(l0, r0)
    st = ti.sin(th0)

    for i, j in states:
        u = (2 * (j + 0.5) / height - 1) * half_h
        v = (2 * (i + 0.5) / width - 1) * half_w

        inorm = 1.0 / ti.sqrt(1 + u * u + v * v)

        vl0 = -1.0 * inorm
        vth0 = u * inorm / r
        vph0 = v * inorm / (r * st)

        states[i, j] = ti.Vector([l0, th0, ph0, vl0, vth0, vph0], dt=ti.f32)


@ti.kernel
def run_simulation_kernel(h: float, hmax: float, r0: float, states: Field, geodesics: Field):
    width = ti.static(states.shape[0])
    imin = int(width // 2 - 5)
    imax = int(width // 2 + 5)
    for i, j in states:
        _h = ti.select(imin <= i <= imax, 1e-3, h)
        geodesics[i, j] = trace_geodesic(states[i, j], _h, hmax, r0)


@ti.func
def sample_texture(texture: Field, th: float, ph: float) -> float:
    H = ti.static(texture.shape[0])
    W = ti.static(texture.shape[1])
    two_pi = 2.0 * ti.math.pi

    # mod phi as it loops around the globe, so phi=0 <=> phi=2pi
    ph = ti.math.mod(ph, two_pi)
    ph = ph + two_pi if ph < 0 else ph
    # do not mod theta because theta=0 <!=> theta=pi, clamp to ensure normalization works
    th = ti.math.clamp(th, 0.0, ti.math.pi)

    x = (ph / two_pi) * (W - 1.0)
    y = (th / ti.math.pi) * (H - 1.0)

    ix = int(ti.math.clamp(x, 0, W - 1))
    iy = int(ti.math.clamp(y, 0, H - 1))

    return texture[iy, ix]


@ti.kernel
def render_image_kernel(geodesics: Field, output: Field, space1: Field, space2: Field):
    for i, j in output:
        l = geodesics[i, j][0]
        th = geodesics[i, j][1]
        ph = geodesics[i, j][2]

        # sample from a different image based on where the ray ended up
        output[i, j] = ti.select(
            l > 0.0,
            ti.cast(sample_texture(space1, th, ph), ti.f32),
            ti.cast(sample_texture(space2, th, ph), ti.f32)
        )


def load_texture(filename: str) -> Field:
    image = ti.tools.imread(filename, channels=3)
    image = image.astype(dtype=np.float32) / 255.0
    field = ti.Vector.field(3, dtype=ti.f32, shape=image.shape[:2])
    field.from_numpy(image)
    return field


def main():
    # width, height = 1980, 1080
    # width, height = 960, 540
    # width, height = 640, 360
    width, height = 426, 240

    fov = 100 * ti.math.pi / 180.0
    h = 1e-2
    hmax = 30.0
    r0 = 1.0
    l0 = 4.0
    th0 = ti.math.pi / 2.0
    ph0 = 0.0

    states = ti.Vector.field(6, dtype=ti.f32, shape=(width, height))
    geodesics = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))
    output = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))
    space1 = load_texture("images/space1.jpg")
    space2 = load_texture("images/space2.jpg")

    gui = ti.GUI("Wormhole", res=(width, height))
    xs, ys, xe, ye = 0, 0, 0, 0
    while gui.running:
        gui.get_event()
        if gui.is_pressed(ti.GUI.LMB):
            xe, ye = gui.get_cursor_pos()
            if xs == ys == 0:
                xs, ys = xe, ye
            ph0 += 2 * np.atan2(xs - xe, 2 * l0) * 10
            th0 += 2 * np.atan2(ys - ye, 2 * l0) * 10
            ph0 = np.mod(ph0, 2 * np.pi)
            th0 = np.mod(th0, np.pi)
            xs, ys = xe, ye
        else:
            xs, ys, xe, ye = 0, 0, 0, 0

        make_initial_states_kernel(
            l0, th0, ph0, r0, fov, height, width, states)
        run_simulation_kernel(h, hmax, r0, states, geodesics)
        render_image_kernel(geodesics, output, space1, space2)
        gui.set_image(output)
        gui.show()


if __name__ == '__main__':
    main()
