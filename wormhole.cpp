#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "cimg.h"
#include "pool.h"

using namespace cimg_library;
typedef CImg<unsigned char> Img;

constexpr float PI = 3.14159265358979323846f;
constexpr float TWO_PI = 2.0f * PI;
constexpr float INV_TWO_PI = 1.0f / TWO_PI;

typedef struct
{
    float l;
    float th;
    float ph;
    float vl;
    float vth;
    float vph;
} State;

inline static float r_of_l(float l, float b, float L)
{
    return powf(powf(l, L) + powf(b, L), 1 / L);
}

inline static float r_prime_of_l(float l, float b, float L)
{
    return powf(powf(l, L) + powf(b, L), 1 / L - 1) * powf(l, L - 1);
}

// Right-hand side of differential equation
void rhs(const State &state, float b, float L, State &result)
{
    float st, ct;
    __sincosf(state.th, &st, &ct);

    float r = r_of_l(state.l, b, L);
    float rp = r_prime_of_l(state.l, b, L);
    float rpr = rp / r;

    float al = r * rp * state.vth * state.vth + r * rp * st * st * state.vph * state.vph;
    float ath = -2 * rpr * state.vl * state.vth + st * ct * state.vph * state.vph;
    float aph = -2 * rpr * state.vl * state.vph - 2 * (ct / (st + 1e-12)) * state.vth * state.vph;

    result.l = state.vl;
    result.th = state.vth;
    result.ph = state.vph;
    result.vl = al;
    result.vth = ath;
    result.vph = aph;
}

// Runge-Kutta 4th order step
void rk4_step(State &s, float dt, float b, float L)
{
    State k1, k2, k3, k4, tmp;

    rhs(s, b, L, k1);

    tmp.l = s.l + 0.5f * dt * k1.l;
    tmp.th = s.th + 0.5f * dt * k1.th;
    tmp.ph = s.ph + 0.5f * dt * k1.ph;
    tmp.vl = s.vl + 0.5f * dt * k1.vl;
    tmp.vth = s.vth + 0.5f * dt * k1.vth;
    tmp.vph = s.vph + 0.5f * dt * k1.vph;
    rhs(tmp, b, L, k2);

    tmp.l = s.l + 0.5f * dt * k2.l;
    tmp.th = s.th + 0.5f * dt * k2.th;
    tmp.ph = s.ph + 0.5f * dt * k2.ph;
    tmp.vl = s.vl + 0.5f * dt * k2.vl;
    tmp.vth = s.vth + 0.5f * dt * k2.vth;
    tmp.vph = s.vph + 0.5f * dt * k2.vph;
    rhs(tmp, b, L, k3);

    tmp.l = s.l + dt * k3.l;
    tmp.th = s.th + dt * k3.th;
    tmp.ph = s.ph + dt * k3.ph;
    tmp.vl = s.vl + dt * k3.vl;
    tmp.vth = s.vth + dt * k3.vth;
    tmp.vph = s.vph + dt * k3.vph;
    rhs(tmp, b, L, k4);

    const float w = dt / 6.0f;
    s.l = s.l + w * (k1.l + 2.f * k2.l + 2.f * k3.l + k4.l);
    s.th = s.th + w * (k1.th + 2.f * k2.th + 2.f * k3.th + k4.th);
    s.ph = s.ph + w * (k1.ph + 2.f * k2.ph + 2.f * k3.ph + k4.ph);
    s.vl = s.vl + w * (k1.vl + 2.f * k2.vl + 2.f * k3.vl + k4.vl);
    s.vth = s.vth + w * (k1.vth + 2.f * k2.vth + 2.f * k3.vth + k4.vth);
    s.vph = s.vph + w * (k1.vph + 2.f * k2.vph + 2.f * k3.vph + k4.vph);
}

inline static int rgb(unsigned char red, unsigned char green, unsigned char blue)
{
    return (red << 16) | (green << 8) | blue;
}

inline static int pixel(const Img &image, const int x, const int y)
{
    unsigned char red = image(x, y, 0, 0);
    unsigned char green = image(x, y, 0, 1);
    unsigned char blue = image(x, y, 0, 2);
    return rgb(red, green, blue);
}

inline static unsigned char red(int rgb)
{
    return (rgb >> 16) & 0xff;
}

inline static unsigned char green(int rgb)
{
    return (rgb >> 8) & 0xff;
}

inline static unsigned char blue(int rgb)
{
    return rgb & 0xff;
}

inline static float clampf(float value, float min, float max)
{
    return fmaxf(min, fminf(value, max));
}

inline float mod2pi(float x)
{
    x -= TWO_PI * floorf(x * INV_TWO_PI);
    return x;
}

int map_coordinates_to_pixel(
    float l, float theta, float phi,
    const Img &space1, const Img &space2)
{
    const auto &space = l > 0 ? space1 : space2;
    const int width = space.width();
    const int height = space.height();

    phi = mod2pi(phi);
    theta = clampf(theta, 0.0, PI);

    float x = (phi / TWO_PI) * (width - 1.0);
    float y = (theta / PI) * (height - 1.0);

    int ix = (int)clampf(x, 0, width - 1);
    int iy = (int)clampf(y, 0, height - 1);
    return pixel(space, ix, iy);
}

// Trace geodesic and return a color value
int trace_geodesic(State &state, float dt, int tmax, float b, float L, const Img &space1, const Img &space2)
{
    int steps = tmax / dt;
    for (int i = 0; i < steps; i++)
    {
        if (fabsf(state.vph) > 0.3 || fabsf(state.vth) > 0.6)
        {
            rk4_step(state, dt, b, L);
        }
        else
        {
            rk4_step(state, 0.1, b, L);
        }

        if (fabsf(state.l) > 4)
        {
            break;
        }
    }
    return map_coordinates_to_pixel(state.l, state.th, state.ph, space1, space2);
}

// Makes the initial state for RK4 integration
void init_state(
    float l0, float th0, float ph0,
    float b, float L,
    float c_r, float c_th, float c_ph,
    State &state)
{
    float r = r_of_l(l0, b, L);
    float st = fmaxf(sin(th0), 1e-9);

    float vl = c_r;
    float vth = c_th / r;
    float vph = c_ph / (r * st);

    state.l = l0;
    state.th = th0;
    state.ph = ph0;
    state.vl = vl;
    state.vth = vth;
    state.vph = vph;
}

// Build a normalized view ray in local basis and return its components along e_r, e_th, e_ph
void pixel_to_direction(
    int i, int j, int W, int H, float fov,
    const float e_r[3], const float e_th[3], const float e_ph[3],
    const float cam_r[3], const float cam_th[3], const float cam_ph[3],
    float &c_r, float &c_th, float &c_ph)
{
    float height = 2.0 * tan(fov / 2.0);
    float width = height * ((float)W / H);

    // pixel to normalized screen offsets (u: up, v: right)
    float u = (1 - 2 * (i + 0.5) / H) * height;
    float v = (2 * (j + 0.5) / W - 1) * width;

    // ray in wormhole coordinates
    float d_r = cam_r[0] + u * cam_th[0] + v * cam_ph[0];
    float d_th = cam_r[1] + u * cam_th[1] + v * cam_ph[1];
    float d_ph = cam_r[2] + u * cam_th[2] + v * cam_ph[2];

    // normalize direction vector
    float norm = sqrt(d_r * d_r + d_th * d_th + d_ph * d_ph);
    d_r /= norm;
    d_th /= norm;
    d_ph /= norm;

    // components in the local orthonormal frame
    c_r = d_r * e_r[0] + d_th * e_r[1] + d_ph * e_r[2];
    c_th = d_r * e_th[0] + d_th * e_th[1] + d_ph * e_th[2];
    c_ph = d_r * e_ph[0] + d_th * e_ph[1] + d_ph * e_ph[2];
}

void init_world_basis(float th0, float ph0, float *e_r, float *e_th, float *e_ph)
{
    float st, ct, sp, cp;
    __sincosf(th0, &st, &ct);
    __sincosf(ph0, &sp, &cp);
    e_r[0] = st * cp;
    e_r[1] = st * sp;
    e_r[2] = ct;
    e_th[0] = ct * cp;
    e_th[1] = ct * sp;
    e_th[2] = -st;
    e_ph[0] = -sp;
    e_ph[1] = cp;
    e_ph[2] = 0.0;
}

void init_camera_basis(
    float *e_r, float *e_th, float *e_ph,
    float *cam_r, float *cam_th, float *cam_ph,
    float r_angle, float th_angle, float ph_angle)
{
    float sr, cr;
    float sth, cth;
    float sph, cph;
    __sincosf(r_angle, &sr, &cr);
    __sincosf(th_angle, &sth, &cth);
    __sincosf(ph_angle, &sph, &cph);

    cam_r[0] = -e_r[0];
    cam_r[1] = -e_r[1];
    cam_r[2] = -e_r[2];

    cam_th[0] = -e_th[0];
    cam_th[1] = -e_th[1];
    cam_th[2] = -e_th[2];

    cam_ph[0] = e_ph[0];
    cam_ph[1] = e_ph[1];
    cam_ph[2] = e_ph[2];

    // Rotation around cam_ph axis
    float temp_r[3], temp_th[3];
    temp_r[0] = cph * cam_r[0] + sph * cam_th[0];
    temp_r[1] = cph * cam_r[1] + sph * cam_th[1];
    temp_r[2] = cph * cam_r[2] + sph * cam_th[2];

    temp_th[0] = sph * cam_r[0] - cph * cam_th[0];
    temp_th[1] = sph * cam_r[1] - cph * cam_th[1];
    temp_th[2] = sph * cam_r[2] - cph * cam_th[2];

    // Rotation around cam_th axis
    cam_r[0] = cth * temp_r[0] + sth * cam_ph[0];
    cam_r[1] = cth * temp_r[1] + sth * cam_ph[1];
    cam_r[2] = cth * temp_r[2] + sth * cam_ph[2];

    cam_ph[0] = -sth * temp_r[0] + cth * cam_ph[0];
    cam_ph[1] = -sth * temp_r[1] + cth * cam_ph[1];
    cam_ph[2] = -sth * temp_r[2] + cth * cam_ph[2];

    // Rotation around cam_r axis
    cam_th[0] = cr * temp_th[0] - sr * cam_ph[0];
    cam_th[1] = cr * temp_th[1] - sr * cam_ph[1];
    cam_th[2] = cr * temp_th[2] - sr * cam_ph[2];

    cam_ph[0] = sr * temp_th[0] + cr * cam_ph[0];
    cam_ph[1] = sr * temp_th[1] + cr * cam_ph[1];
    cam_ph[2] = sr * temp_th[2] + cr * cam_ph[2];
}

void render_row(
    const Img &space1, const Img &space2, Img &output,
    int W, int H, int row,
    float fov, float b, float L, float dt, float tmax,
    float l0, float th0, float ph0,
    float r_ang, float th_ang, float ph_ang)
{
    float e_r[3], e_th[3], e_ph[3];
    float cam_r[3], cam_th[3], cam_ph[3];

    State state;
    float c_r, c_th, c_ph;

    init_world_basis(th0, ph0, e_r, e_th, e_ph);
    init_camera_basis(e_r, e_th, e_ph, cam_r, cam_th, cam_ph, r_ang, th_ang, ph_ang);

    for (int j = 0; j < W; j++)
    {
        pixel_to_direction(row, j, W, H, fov, e_r, e_th, e_ph, cam_r, cam_th, cam_ph, c_r, c_th, c_ph);
        init_state(l0, th0, ph0, b, L, c_r, c_th, c_ph, state);
        int rgb = trace_geodesic(state, dt, tmax, b, L, space1, space2);
        output(j, row, 0, 0) = red(rgb);
        output(j, row, 0, 1) = green(rgb);
        output(j, row, 0, 2) = blue(rgb);
    }
}

void render_image(
    const Img &space1, const Img &space2, std::string filename,
    ThreadPool &pool,
    int W, int H,
    float fov, float b, float L, float dt, float tmax,
    float l0, float th0, float ph0,
    float r_ang, float th_ang, float ph_ang)
{
    Img output(W, H, 1, 3, 0);
    for (int i = 0; i < H; i++)
    {
        pool.enqueue(
            &render_row,
            std::ref(space1), std::ref(space2), std::ref(output),
            W, H, i, fov, b, L, dt, tmax, l0, th0, ph0, r_ang, th_ang, ph_ang);
    }
    pool.wait_idle();
    output.save_png(filename.c_str());
}

inline static float clamp_norm(float x, float xmin, float xmax)
{
    if (x < xmin)
    {
        return 0;
    }
    if (x > xmax)
    {
        return 1;
    }
    return (x - xmin) / (xmax - xmin);
}

inline static float lerp(float x, float xmin, float xmax, float ymin, float ymax)
{
    float n = clamp_norm(x, xmin, xmax);
    return ymin + n * (ymax - ymin);
}

inline static float aderp(float x, float xmin, float xmax, float ymin, float ymax)
{
    float n = clamp_norm(x, xmin, xmax);
    float r = (cos((n + 1) * PI) + 1) / 2;
    return ymin + r * (ymax - ymin);
}

int main()
{
    Img space1("images/space1.jpg");
    Img space2("images/space2.jpg");
    ThreadPool pool(9);

    const int W = 160, H = 90;
    // const int W = 320, H = 180;
    // const int W = 640, H = 360;
    // const int W = 1280, H = 720;
    // const int W = 1920, H = 1080;
    // const int W = 3840, H = 2160;

    const float fov = 60 * PI / 180;
    const float b = 1;
    const float L = 4;
    const float dt = 1e-1; // 1e-3 to avoid artefacts
    const float tmax = 20.0;

    // relative to world coordinates
    // const float l0 = 3;
    const float th0 = PI / 2;
    // const float ph0 = 0;

    // relative to camera coordinates
    const float r_ang = 0;
    // const float th_ang = 0;
    const float ph_ang = 0;

    const int fps = 24;
    const int duration = 45;
    const int frames = fps * duration;
    const int hframes = frames / 2;

    for (int i = 0; i < frames; i++)
    {
        float l0 = aderp(i, 0, frames - 1, 3, -3);
        float ph0 = aderp(i, 0, frames - 1, 0, 4 * PI);
        float th_ang = aderp(i, hframes - 8 * fps, hframes + 8 * fps, 0, PI);

        std::string filename = "tmp/video7/wormhole_" + std::to_string(i) + ".png";
        render_image(
            space1, space2, filename, pool,
            W, H, fov, b, L, dt, tmax,
            l0, th0, ph0, r_ang, th_ang, ph_ang);
        printf("%d/%d\n", i + 1, frames);
    }
}