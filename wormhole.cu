#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define CUDA_CHECK(err)                                                      \
    {                                                                        \
        cudaError_t err_code = err;                                          \
        if (err_code != cudaSuccess)                                         \
        {                                                                    \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err_code)      \
                      << " in " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    }

#define PI 3.14159265358979323846f
#define TWO_PI 6.28318530717958647692f
#define INV_TWO_PI 0.15915494309189533577f

__forceinline__ __device__ float clampf(float x, float a, float b)
{
    return fminf(b, fmaxf(a, x));
}

__forceinline__ __device__ float mod2pi(float phi)
{
    phi -= floorf(phi * INV_TWO_PI) * TWO_PI;
    return phi;
}

struct State
{
    float l, th, ph;
    float vl, vth, vph;
};

__forceinline__ __device__ float r_of_l(float l, float b, float L)
{
    return powf(powf(l, L) + powf(b, L), 1 / L);
}

__forceinline__ __device__ float r_prime_of_l(float l, float b, float L)
{
    return powf(powf(l, L) + powf(b, L), 1 / L - 1) * powf(l, L - 1);
}

__forceinline__ __device__ void rhs(const State &s, float b, float L, State &out)
{
    float st, ct;
    __sincosf(s.th, &st, &ct);

    float r = r_of_l(s.l, b, L);
    float rp = r_prime_of_l(s.l, b, L);
    float rpr = rp / (r + 1e-12f);

    float al = r * rp * s.vth * s.vth + r * rp * st * st * s.vph * s.vph;
    float ath = -2 * rpr * s.vl * s.vth + st * ct * s.vph * s.vph;
    float aph = -2 * rpr * s.vl * s.vph - 2 * (ct / (st + 1e-12f)) * s.vth * s.vph;

    out.l = s.vl;
    out.th = s.vth;
    out.ph = s.vph;
    out.vl = al;
    out.vth = ath;
    out.vph = aph;
}

__forceinline__ __device__ void rk4_step(State &s, float dt, float b, float L)
{
    State k1, k2, k3, k4, tmp;

    rhs(s, b, L, k1);

    const float hdt = 0.5f * dt;
    tmp.l = __fmaf_rn(hdt, k1.l, s.l);
    tmp.th = __fmaf_rn(hdt, k1.th, s.th);
    tmp.ph = __fmaf_rn(hdt, k1.ph, s.ph);
    tmp.vl = __fmaf_rn(hdt, k1.vl, s.vl);
    tmp.vth = __fmaf_rn(hdt, k1.vth, s.vth);
    tmp.vph = __fmaf_rn(hdt, k1.vph, s.vph);
    rhs(tmp, b, L, k2);

    tmp.l = __fmaf_rn(hdt, k2.l, s.l);
    tmp.th = __fmaf_rn(hdt, k2.th, s.th);
    tmp.ph = __fmaf_rn(hdt, k2.ph, s.ph);
    tmp.vl = __fmaf_rn(hdt, k2.vl, s.vl);
    tmp.vth = __fmaf_rn(hdt, k2.vth, s.vth);
    tmp.vph = __fmaf_rn(hdt, k2.vph, s.vph);
    rhs(tmp, b, L, k3);

    tmp.l = __fmaf_rn(dt, k3.l, s.l);
    tmp.th = __fmaf_rn(dt, k3.th, s.th);
    tmp.ph = __fmaf_rn(dt, k3.ph, s.ph);
    tmp.vl = __fmaf_rn(dt, k3.vl, s.vl);
    tmp.vth = __fmaf_rn(dt, k3.vth, s.vth);
    tmp.vph = __fmaf_rn(dt, k3.vph, s.vph);
    rhs(tmp, b, L, k4);

    const float w = dt / 6.0f;
    const float w2 = w * 2.0f;

    s.l = __fmaf_rn(w, k1.l, __fmaf_rn(w2, k2.l + k3.l, __fmaf_rn(w, k4.l, s.l)));
    s.th = __fmaf_rn(w, k1.th, __fmaf_rn(w2, k2.th + k3.th, __fmaf_rn(w, k4.th, s.th)));
    s.ph = __fmaf_rn(w, k1.ph, __fmaf_rn(w2, k2.ph + k3.ph, __fmaf_rn(w, k4.ph, s.ph)));
    s.vl = __fmaf_rn(w, k1.vl, __fmaf_rn(w2, k2.vl + k3.vl, __fmaf_rn(w, k4.vl, s.vl)));
    s.vth = __fmaf_rn(w, k1.vth, __fmaf_rn(w2, k2.vth + k3.vth, __fmaf_rn(w, k4.vth, s.vth)));
    s.vph = __fmaf_rn(w, k1.vph, __fmaf_rn(w2, k2.vph + k3.vph, __fmaf_rn(w, k4.vph, s.vph)));
}

__forceinline__ __device__ void init_world_basis(float th0, float ph0, float3 &e_r, float3 &e_th, float3 &e_ph)
{
    float st, ct, sp, cp;
    __sincosf(th0, &st, &ct);
    __sincosf(ph0, &sp, &cp);
    e_r = make_float3(st * cp, st * sp, ct);
    e_th = make_float3(ct * cp, ct * sp, -st);
    e_ph = make_float3(-sp, cp, 0.0f);
}

__forceinline__ __device__ void init_camera_basis(
    float3 &e_r, float3 &e_th, float3 &e_ph,
    float3 &cam_r, float3 &cam_th, float3 &cam_ph,
    float r_angle, float th_angle, float ph_angle)
{
    float sr, cr, sth, cth, sph, cph;
    __sincosf(r_angle, &sr, &cr);
    __sincosf(th_angle, &sth, &cth);
    __sincosf(ph_angle, &sph, &cph);

    cam_r = make_float3(-e_r.x, -e_r.y, -e_r.z);
    cam_th = make_float3(-e_th.x, -e_th.y, -e_th.z);
    cam_ph = e_ph;

    // Rotation around cam_ph axis
    float3 temp_r = make_float3(
        cph * cam_r.x + sph * cam_th.x,
        cph * cam_r.y + sph * cam_th.y,
        cph * cam_r.z + sph * cam_th.z);

    float3 temp_th = make_float3(
        sph * cam_r.x - cph * cam_th.x,
        sph * cam_r.y - cph * cam_th.y,
        sph * cam_r.z - cph * cam_th.z);

    // Rotation around cam_th axis
    cam_r = make_float3(
        cth * temp_r.x + sth * cam_ph.x,
        cth * temp_r.y + sth * cam_ph.y,
        cth * temp_r.z + sth * cam_ph.z);

    cam_ph = make_float3(
        -sth * temp_r.x + cth * cam_ph.x,
        -sth * temp_r.y + cth * cam_ph.y,
        -sth * temp_r.z + cth * cam_ph.z);

    // Rotation around cam_r axis
    cam_th = make_float3(
        cr * temp_th.x - sr * cam_ph.x,
        cr * temp_th.y - sr * cam_ph.y,
        cr * temp_th.z - sr * cam_ph.z);

    cam_ph = make_float3(
        sr * temp_th.x + cr * cam_ph.x,
        sr * temp_th.y + cr * cam_ph.y,
        sr * temp_th.z + cr * cam_ph.z);
}

__forceinline__ __device__ void pixel_to_direction(
    int i, int j, int W, int H, float tanHalfFov, float aspect,
    const float3 &e_r, const float3 &e_th, const float3 &e_ph,
    const float3 &cam_r, const float3 &cam_th, const float3 &cam_ph,
    float &c_r, float &c_th, float &c_ph)
{
    float height = 2.0f * tanHalfFov;
    float width = height * aspect;

    float u = (1.0f - 2.0f * (i + 0.5f) / H) * height;
    float v = (2.0f * (j + 0.5f) / W - 1.0f) * width;

    float3 d = make_float3(
        cam_r.x + u * cam_th.x + v * cam_ph.x,
        cam_r.y + u * cam_th.y + v * cam_ph.y,
        cam_r.z + u * cam_th.z + v * cam_ph.z);

    float invn = rsqrtf(d.x * d.x + d.y * d.y + d.z * d.z);
    d.x *= invn;
    d.y *= invn;
    d.z *= invn;

    c_r = d.x * e_r.x + d.y * e_r.y + d.z * e_r.z;
    c_th = d.x * e_th.x + d.y * e_th.y + d.z * e_th.z;
    c_ph = d.x * e_ph.x + d.y * e_ph.y + d.z * e_ph.z;
}

__forceinline__ __device__ uchar3 get_rgb_tex(cudaTextureObject_t tex, int width, float x, float y)
{
    uchar4 val = tex2D<uchar4>(tex, x + 0.5f, y + 0.5f);
    return make_uchar3(val.x, val.y, val.z);
}

__forceinline__ __device__ uchar3 map_coordinates_to_pixel(
    float l, float theta, float phi,
    cudaTextureObject_t space1, int width1, int height1,
    cudaTextureObject_t space2, int width2, int height2)
{
    cudaTextureObject_t space = (l > 0) ? space1 : space2;
    int width = (l > 0) ? width1 : width2;
    int height = (l > 0) ? height1 : height2;

    phi = mod2pi(phi);
    theta = clampf(theta, 0, PI);

    float xf = (phi * (0.5f / PI)) * (float)(width - 1);
    float yf = (theta * (1.0f / PI)) * (float)(height - 1);

    return get_rgb_tex(space, width, xf, yf);
}

__forceinline__ __device__ void init_state(
    float l0, float th0, float ph0,
    float b, float L,
    float c_r, float c_th, float c_ph,
    State &s)
{
    float r = r_of_l(l0, b, L);
    float st = fmaxf(sinf(th0), 1e-9f);

    float vl = c_r;
    float vth = c_th / r;
    float vph = c_ph / (r * st);

    s = {l0, th0, ph0, vl, vth, vph};
}

__forceinline__ __device__ uchar3 trace_geodesic(
    State &s, float dt, float tmax, float b, float L,
    cudaTextureObject_t space1, int w1, int h1,
    cudaTextureObject_t space2, int w2, int h2)
{
    const int steps = __float2int_rn(tmax / dt);

#pragma unroll 4
    for (int i = 0; i < steps; i++)
    {
        rk4_step(s, dt, b, L);
        if (s.l * s.l > 16)
        {
            break;
        }
    }

    return map_coordinates_to_pixel(s.l, s.th, s.ph, space1, w1, h1, space2, w2, h2);
}

#ifndef BLOCK_DIM
#define BLOCK_DIM 256
#endif

__launch_bounds__(BLOCK_DIM, 2)
    __global__ void wormhole_kernel(
        uchar3 *__restrict__ output,
        int W, int H,
        cudaTextureObject_t space1, int w1, int h1,
        cudaTextureObject_t space2, int w2, int h2,
        float fov, float b, float L, float dt, float tmax,
        float l0, float th0, float ph0,
        float r_ang, float th_ang, float ph_ang)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    const float tanHalf = __tanf(0.5f * fov);
    const float aspect = (float)W / H;

    float3 e_r, e_th, e_ph, cam_r, cam_th, cam_ph;
    init_world_basis(th0, ph0, e_r, e_th, e_ph);
    init_camera_basis(e_r, e_th, e_ph, cam_r, cam_th, cam_ph, r_ang, th_ang, ph_ang);

    for (int idx = tid; idx < W * H; idx += stride)
    {
        int i = idx / W;
        int j = idx % W;

        float c_r, c_th, c_ph;
        pixel_to_direction(i, j, W, H, tanHalf, aspect, e_r, e_th, e_ph, cam_r, cam_th, cam_ph, c_r, c_th, c_ph);

        State s;
        init_state(l0, th0, ph0, b, L, c_r, c_th, c_ph, s);

        output[idx] = trace_geodesic(s, dt, tmax, b, L, space1, w1, h1, space2, w2, h2);
    }
}

cudaTextureObject_t createTexture(const unsigned char *data, int width, int height)
{
    cudaArray_t cuArray;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    CUDA_CHECK(cudaMallocArray(&cuArray, &channelDesc, width, height));

    // Convert RGB to RGBA
    size_t pitch = width * 4;
    unsigned char *rgba = (unsigned char *)malloc(height * pitch);
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            int src_idx = 3 * (i * width + j);
            int dst_idx = 4 * (i * width + j);
            rgba[dst_idx + 0] = data[src_idx + 0];
            rgba[dst_idx + 1] = data[src_idx + 1];
            rgba[dst_idx + 2] = data[src_idx + 2];
            rgba[dst_idx + 3] = 255;
        }
    }

    CUDA_CHECK(cudaMemcpy2DToArray(cuArray, 0, 0, rgba, pitch, width * 4, height, cudaMemcpyHostToDevice));
    free(rgba);

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    cudaTextureObject_t texObj = 0;
    CUDA_CHECK(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr));

    return texObj;
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

inline static float aderp(float x, float xmin, float xmax, float ymin, float ymax)
{
    float n = clamp_norm(x, xmin, xmax);
    float r = (cos((n + 1) * PI) + 1) / 2;
    return ymin + r * (ymax - ymin);
}

void trace_wormhole(
    unsigned char *output, int W, int H,
    const unsigned char *space1, int w1, int h1,
    const unsigned char *space2, int w2, int h2)
{
    uchar3 *d_out = nullptr;
    const size_t out_size = (size_t)W * H * sizeof(uchar3);

    CUDA_CHECK(cudaMalloc(&d_out, out_size));

    cudaTextureObject_t tex1 = createTexture(space1, w1, h1);
    cudaTextureObject_t tex2 = createTexture(space2, w2, h2);

    int deviceId;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&deviceId));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));

    const int N = W * H;
    const int block = BLOCK_DIM;
    int grid = (N + block - 1) / block;
    grid = min(grid, prop.multiProcessorCount * 4);

    const float fov = 60 * PI / 180;
    const float b = 1;
    const float L = 4;
    const float dt = 1e-3f;
    const float tmax = 20;

    // const float l0 = 3;
    const float th0 = PI / 2;
    // const float ph0 = 0;

    const float r_ang = 0;
    // const float th_ang = 0;
    const float ph_ang = 0;

    const int fps = 30;
    const int duration = 45;
    const int frames = duration * fps;
    const int hframes = frames / 2;

    uchar3 *host_out = (uchar3 *)malloc(out_size);

    for (int i = 0; i < frames; i++)
    {
        float l0 = aderp(i, 0, frames - 1, 3, -3);
        float ph0 = aderp(i, 0, frames - 1, 0, 4 * PI);
        float th_ang = aderp(i, hframes - 8 * fps, hframes + 8 * fps, 0, PI);

        wormhole_kernel<<<grid, block>>>(
            d_out, W, H, tex1, w1, h1, tex2, w2, h2,
            fov, b, L, dt, tmax, l0, th0, ph0,
            r_ang, th_ang, ph_ang);

        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(host_out, d_out, out_size, cudaMemcpyDeviceToHost));

        // Convert uchar3 to packed RGB
        for (int p = 0; p < W * H; p++)
        {
            output[3 * p + 0] = host_out[p].x;
            output[3 * p + 1] = host_out[p].y;
            output[3 * p + 2] = host_out[p].z;
        }

        std::string filename = "/content/drive/MyDrive/output/wormhole_" + std::to_string(i) + ".png";
        int ok = stbi_write_png(filename.c_str(), W, H, 3, output, W * 3);
        if (ok)
            std::cout << i << "/" << frames << std::endl;
        else
            std::cerr << "Error: Failed to save image.\n";
    }

    free(host_out);
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaDestroyTextureObject(tex1));
    CUDA_CHECK(cudaDestroyTextureObject(tex2));
}

int main()
{
    int w1, h1, w2, h2;
    int outW = 3840;
    int outH = 2160;

    int comp1 = 0, comp2 = 0;
    unsigned char *space1 = stbi_load("/content/drive/MyDrive/spaces/space3.jpg", &w1, &h1, &comp1, 3);
    unsigned char *space2 = stbi_load("/content/drive/MyDrive/spaces/space4.jpg", &w2, &h2, &comp2, 3);

    if (!space1 || !space2)
    {
        std::cerr << "Failed to load images." << std::endl;
        return 1;
    }

    unsigned char *output = (unsigned char *)malloc((size_t)outW * outH * 3);

    std::printf("space1: %dx%d\n", w1, h1);
    std::printf("space2: %dx%d\n", w2, h2);
    std::printf("output: %dx%d\n", outW, outH);

    trace_wormhole(output, outW, outH, space1, w1, h1, space2, w2, h2);

    stbi_image_free(space1);
    stbi_image_free(space2);
    free(output);
    return 0;
}