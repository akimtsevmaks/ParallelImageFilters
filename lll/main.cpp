#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define _CRT_SECURE_NO_WARNINGS
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <chrono>
#include <vector>
#include <algorithm>
#include <functional>
#include <cmath>
#include <immintrin.h>
#include <omp.h>
#include <cstring>
#include <iomanip>
#include <iostream>

#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>


using uchar = unsigned char;


static const char* OCL_SRC = R"(
typedef unsigned char uchar;
#define MNMX(a,b) { uchar _l = min(a, b); uchar _h = max(a, b); a = _l; b = _h; }
 
__kernel void k_invert(__global const uchar* src, __global uchar* dst, int w, int h, int ch) {
    int x = get_global_id(0), y = get_global_id(1);
    if(x >= w || y >= h) return;
    int base = (y * w + x) * ch;
    for (int c = 0; c < (ch == 4 ? 3 : ch); c++) dst[base + c] = 255 - src[base + c];
    if (ch == 4) dst[base + 3] = src[base + 3];
}
 
__kernel void k_median(__global const uchar* src, __global uchar* dst, int w, int h, int ch) {
    int x = get_global_id(0), y = get_global_id(1);
    if(x >= w || y >= h) return;
    int base = (y * w + x) * ch;
    if(x == 0 || x == w - 1 || y == 0 || y == h - 1) {
        for(int c = 0; c < ch; c++) dst[base + c] = src[base + c]; return;
    }
    for(int c = 0; c < (ch == 4 ? 3 : ch); c++) {
        uchar p0 = src[((y - 1) * w + (x - 1)) * ch + c], p1 = src[((y - 1) * w + x) * ch + c],
              p2 = src[((y - 1) * w + (x + 1)) * ch + c], p3 = src[(y * w + (x - 1)) * ch + c],
              p4 = src[(y * w + x) * ch + c], p5 = src[(y * w + (x + 1)) * ch + c],
              p6 = src[((y + 1) * w + (x - 1)) * ch + c], p7 = src[((y + 1) * w + x) * ch + c],
              p8 = src[((y + 1) * w + (x + 1)) * ch + c];
        MNMX(p1, p2); MNMX(p4, p5); MNMX(p7, p8);
        MNMX(p0, p1); MNMX(p3, p4); MNMX(p6, p7);
        MNMX(p1, p2); MNMX(p4, p5); MNMX(p7, p8);
        MNMX(p0, p3); MNMX(p5, p8); MNMX(p4, p7);
        MNMX(p3, p6); MNMX(p1, p4); MNMX(p2, p5);
        MNMX(p4, p7); MNMX(p4, p2); MNMX(p6, p4); MNMX(p4, p2);
        dst[base + c] = p4;
    }
    if(ch == 4) dst[base + 3] = src[base + 3];
}
 
__kernel void k_edges(__global const uchar* src, __global uchar* dst, int w, int h, int ch) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    int base = (y * w + x) * ch;
    if (x == 0 || x == w - 1 || y == 0 || y == h - 1) {
        for (int c = 0; c < ch; c++) dst[base + c] = 0;
        if (ch == 4) dst[base + 3] = 255; return;
    }
    #define LUM(px, py) \
        ((src[((py) * w + (px)) * ch] * 77 + src[((py) * w + (px)) * ch + 1] * 150 + src[((py) * w + (px)) * ch + 2] * 29) >> 8)
    int L00 = LUM(x - 1, y - 1), L01 = LUM(x, y - 1), L02 = LUM(x + 1, y - 1);
    int L10 = LUM(x - 1, y), L12 = LUM(x + 1, y);
    int L20 = LUM(x - 1, y + 1), L21 = LUM(x, y + 1), L22 = LUM(x + 1, y + 1);
    int Gx = -L00 + L02 - 2 * L10 + 2 * L12 - L20 + L22;
    int Gy = -L00 - 2 * L01 - L02 + L20 + 2 * L21 + L22;
    int val = abs(Gx) + abs(Gy);
    if (val > 255) val = 255;
    for (int c = 0; c < (ch == 4 ? 3 : ch); c++) dst[base + c] = (uchar)val;
    if (ch == 4) dst[base + 3] = 255;
    #undef LUM
}
#undef MNMX
)";


struct OCLCtx {
	cl_platform_id platform = nullptr;
	cl_device_id device = nullptr;
	cl_context context = nullptr;
    cl_command_queue queue = nullptr;

    cl_program program = nullptr;
    cl_kernel k_invert = nullptr;
    cl_kernel k_median = nullptr;
    cl_kernel k_edges = nullptr;

    bool isValid = false;
};


OCLCtx init_opencl() {
	OCLCtx ctx;
	cl_uint numPlatforms = 0;

    if (clGetPlatformIDs(0, nullptr, &numPlatforms) != CL_SUCCESS || numPlatforms == 0) return ctx;

	std::vector<cl_platform_id> platforms(numPlatforms);
	clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);

	cl_device_id selectedDevice = nullptr; 
	cl_platform_id selectedPlatform = nullptr;

    auto findDevice = [&](cl_device_type type, bool IntelOnly) -> bool {
        for (auto p : platforms) {
            if (IntelOnly) {
                char name[256] = { 0 };
                clGetPlatformInfo(p, CL_PLATFORM_NAME, sizeof(name), name, nullptr);
                if (std::string(name).find("Intel") == std::string::npos) continue;
            }

            cl_uint numDevices = 0;

            if (clGetDeviceIDs(p, type, 0, nullptr, &numDevices) != CL_SUCCESS || numDevices == 0) continue;

            std::vector<cl_device_id> devices(numDevices);
            clGetDeviceIDs(p, type, numDevices, devices.data(), nullptr);

            selectedDevice = devices[0];
            selectedPlatform = p;
            return true;
        }
        return false;
    };

    if (!findDevice(CL_DEVICE_TYPE_GPU, true) && !findDevice(CL_DEVICE_TYPE_CPU, true) &&
        !findDevice(CL_DEVICE_TYPE_GPU, false) && !findDevice(CL_DEVICE_TYPE_CPU, false)) return ctx;
        
	ctx.platform = selectedPlatform;
	ctx.device = selectedDevice;

    cl_int err;
	ctx.context = clCreateContext(nullptr, 1, &ctx.device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) return ctx;

	ctx.queue = clCreateCommandQueueWithProperties(ctx.context, ctx.device, 0, &err);
    if (err != CL_SUCCESS) ctx.queue = clCreateCommandQueue(ctx.context, ctx.device, 0, &err);

    if (err != CL_SUCCESS) {
        clReleaseContext(ctx.context);
        ctx.context = nullptr;
        return ctx;
    }

	ctx.isValid = true;
    return ctx;
}

bool build_ocl_kernels(OCLCtx& ctx) {
    if (!ctx.isValid) return false;

    cl_int err;
    ctx.program = clCreateProgramWithSource(ctx.context, 1, &OCL_SRC, nullptr, &err);
    if (err != CL_SUCCESS) return false;

    err = clBuildProgram(ctx.program, 1, &ctx.device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t sz = 0;
        clGetProgramBuildInfo(ctx.program, ctx.device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &sz);
        std::string log(sz, ' ');
        clGetProgramBuildInfo(ctx.program, ctx.device, CL_PROGRAM_BUILD_LOG, sz, log.data(), nullptr);
        std::cout << "Ошибка:\n" << log << "\n"; return false;
    }

    ctx.k_invert = clCreateKernel(ctx.program, "k_invert", &err);
    ctx.k_median = clCreateKernel(ctx.program, "k_median", &err);
    ctx.k_edges = clCreateKernel(ctx.program, "k_edges", &err);
    return err == CL_SUCCESS;
}

void release_opencl(OCLCtx& ctx) {
    if (ctx.k_edges) clReleaseKernel(ctx.k_edges);
    if (ctx.k_median) clReleaseKernel(ctx.k_median);
    if (ctx.k_invert) clReleaseKernel(ctx.k_invert);
    if (ctx.program) clReleaseProgram(ctx.program);
    if (ctx.queue) clReleaseCommandQueue(ctx.queue);
    if (ctx.context) clReleaseContext(ctx.context);
}

size_t round_up(size_t n, size_t m) { return ((n + m - 1) / m) * m; }


static void run_kernel(cl_kernel kernel, const uchar* src, uchar* dst, int w, int h, int ch, OCLCtx& ctx) {
    size_t imgSize = (size_t)w * h * ch;
    cl_int err;

    cl_mem bufSrc = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, imgSize, (void*)src, &err);
    cl_mem bufDst = clCreateBuffer(ctx.context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, imgSize, dst, &err);
    cl_int W = w, H = h, CH = ch;

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufSrc);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufDst);
    clSetKernelArg(kernel, 2, sizeof(cl_int), &W);
    clSetKernelArg(kernel, 3, sizeof(cl_int), &H);
    clSetKernelArg(kernel, 4, sizeof(cl_int), &CH);

    size_t global[2] = { round_up(w , 16), round_up(h, 16) }, local[2] = { 16, 16 };
    clEnqueueNDRangeKernel(ctx.queue, kernel, 2, nullptr, global, local, 0, nullptr, nullptr);
    clFinish(ctx.queue);
    clReleaseMemObject(bufSrc);
    clReleaseMemObject(bufDst);
}

double measure_ms(std::function<void()> fn, int runs = 5) {
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < runs; i++) {
        fn();
    }
	auto end = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double, std::milli> elapsed = end - start;
	return elapsed.count() / runs;
}


void median_seq(const uchar* src, uchar* dst, int w, int h, int ch) {
    std::memcpy(dst, src, w * h * ch);

    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            for (int c = 0; c < ch; c++) {
                uchar window[9];
                int k = 0;

                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        window[k++] = src[((y + dy) * w + (x + dx)) * ch + c];
                    }
                }

                std::sort(window, window + 9);
				dst[(y * w + x) * ch + c] = window[4];
            }
        }
    }
}

void median_omp(const uchar* src, uchar* dst, int w, int h, int ch) {
    std::memcpy(dst, src, w * h * ch);

    #pragma omp parallel for schedule(dynamic)
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            for (int c = 0; c < ch; c++) {
                uchar window[9];
                int k = 0;

                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        window[k++] = src[((y + dy) * w + (x + dx)) * ch + c];
                    }
                }

                std::sort(window, window + 9);
                dst[(y * w + x) * ch + c] = window[4];
            }
        }
    }
}

void median_simd(const uchar* src, uchar* dst, int w, int h, int ch) {
    std::memcpy(dst, src, (size_t)w * h * ch);
    const int rowBytes = w * ch;

#define MNMX(a, b) { __m256i _lo = _mm256_min_epu8(a, b); \
                     (b) = _mm256_max_epu8(a, b); (a) = _lo; }

    for (int y = 1; y < h - 1; y++) {
        int base = (y * w + 1) * ch;
        int endB = (y * w + (w - 1)) * ch;
        int b = base;

        for (; b + 32 <= endB; b += 32) {
            __m256i p0 = _mm256_loadu_si256((const __m256i*)(src + b - rowBytes - ch));
            __m256i p1 = _mm256_loadu_si256((const __m256i*)(src + b - rowBytes));
            __m256i p2 = _mm256_loadu_si256((const __m256i*)(src + b - rowBytes + ch));
            __m256i p3 = _mm256_loadu_si256((const __m256i*)(src + b - ch));
            __m256i p4 = _mm256_loadu_si256((const __m256i*)(src + b));
            __m256i p5 = _mm256_loadu_si256((const __m256i*)(src + b + ch));
            __m256i p6 = _mm256_loadu_si256((const __m256i*)(src + b + rowBytes - ch));
            __m256i p7 = _mm256_loadu_si256((const __m256i*)(src + b + rowBytes));
            __m256i p8 = _mm256_loadu_si256((const __m256i*)(src + b + rowBytes + ch));

            MNMX(p1, p2); MNMX(p4, p5); MNMX(p7, p8);
            MNMX(p0, p1); MNMX(p3, p4); MNMX(p6, p7);
            MNMX(p1, p2); MNMX(p4, p5); MNMX(p7, p8);
            MNMX(p0, p3); MNMX(p5, p8); MNMX(p4, p7);
            MNMX(p3, p6); MNMX(p1, p4); MNMX(p2, p5);
            MNMX(p4, p7); MNMX(p4, p2); MNMX(p6, p4);
            MNMX(p4, p2);

            _mm256_storeu_si256((__m256i*)(dst + b), p4);
        }

        for (; b < endB; b++) {
            uchar win[9] = {
                src[b - rowBytes - ch], src[b - rowBytes], src[b - rowBytes + ch],
                src[b - ch],            src[b],            src[b + ch],
                src[b + rowBytes - ch], src[b + rowBytes], src[b + rowBytes + ch]
            };
            std::sort(win, win + 9);
            dst[b] = win[4];
        }
    }

#undef MNMX
}

void median_ocl(const uchar* src, uchar* dst, int w, int h, int ch, OCLCtx& ctx) {
    if (!ctx.isValid || !ctx.k_median) { std::memcpy(dst, src, (size_t)w * h * ch); return; }
    run_kernel(ctx.k_median, src, dst, w, h, ch, ctx);
}

void invert_seq(const uchar* src, uchar* dst, int w, int h, int ch) {
	int totalPixels = w * h;
    for (int i = 0; i < totalPixels; i++) {
        for (int c = 0; c < ch; c++) {
            int idx = i * ch + c;

            if (ch == 4 && c == 3) {
                dst[idx] = src[idx];
            }
            else {
				dst[idx] = 255 - src[idx];
            }
        }
    }
}

void invert_omp(const uchar* src, uchar* dst, int w, int h, int ch) {
    int totalPixels = w * h;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < totalPixels; i++) {
        for (int c = 0; c < ch; c++) {
            int idx = i * ch + c;

            if (ch == 4 && c == 3) {
                dst[idx] = src[idx];
            }
            else {
                dst[idx] = 255 - src[idx];
            }
        }
    }
}

void invert_simd(const uchar* src, uchar* dst, int w, int h, int ch) {
    size_t total = (size_t)w * h * ch;
    size_t i = 0;

    if (ch == 3) {
        __m256i ones = _mm256_set1_epi8((char)0xFF);
        for (; i + 32 <= total; i += 32) {
            __m256i chunk = _mm256_loadu_si256((const __m256i*)(src + i));
            __m256i result = _mm256_sub_epi8(ones, chunk);
			_mm256_storeu_si256((__m256i*)(dst + i), result);
        }
    }
    else {
        __m256i ones_mask = _mm256_set_epi8(
            0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1,
            0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1);
		__m256i ones = _mm256_set1_epi8((char)0xFF);
        for (; i + 32 <= total; i += 32) {
            __m256i chunk = _mm256_loadu_si256((const __m256i*)(src + i));
            __m256i inv = _mm256_sub_epi8(ones, chunk);
            __m256i result = _mm256_or_si256(
                _mm256_and_si256(ones_mask, inv),
                _mm256_andnot_si256(ones_mask, chunk));
            _mm256_storeu_si256((__m256i*)(dst + i), result);
        }
    }

    for (; i < total; i++) {
		if (ch == 4 && (i % 4) == 3) {
			dst[i] = src[i];
		}
		else {
			dst[i] = 255 - src[i];
		}
    }
}

void invert_ocl(const uchar* src, uchar* dst, int w, int h, int ch, OCLCtx& ctx) {
    if (!ctx.isValid || !ctx.k_invert) { std::memcpy(dst, src, (size_t)w * h * ch); return; }
    run_kernel(ctx.k_invert, src, dst, w, h, ch, ctx);
}

void edges_seq(const uchar* src, uchar* dst, int w, int h, int ch) {
    auto get_luminance = [&](int x, int y) -> int {
        const uchar* p = &src[(y * w + x) * ch];
        return (p[0] * 77 + p[1] * 150 + p[2] * 29) >> 8;
    };

    for (int i = 0; i < w * h; i++) {
        dst[i * ch + 0] = 0;
        dst[i * ch + 1] = 0;
		dst[i * ch + 2] = 0;
        if (ch == 4) dst[i * ch + 3] = 255;
    }

    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            int L00 = get_luminance(x - 1, y - 1);
			int L01 = get_luminance(x, y - 1);
			int L02 = get_luminance(x + 1, y - 1);

			int L10 = get_luminance(x - 1, y);
			int L12 = get_luminance(x + 1, y);

			int L20 = get_luminance(x - 1, y + 1);
			int L21 = get_luminance(x, y + 1);
			int L22 = get_luminance(x + 1, y + 1);

			int Gx = -L00 + L02 - 2 * L10 + 2 * L12 - L20 + L22;
			int Gy = -L00 - 2 * L01 - L02 + L20 + 2 * L21 + L22;

            int val = std::abs(Gx) + std::abs(Gy);
			if (val > 255) val = 255;

			int idx = (y * w + x) * ch;
            dst[idx + 0] = val;
			dst[idx + 1] = val;
			dst[idx + 2] = val;
        }
    }
}

void edges_omp(const uchar* src, uchar* dst, int w, int h, int ch) {
    auto get_luminance = [&](int x, int y) -> int {
        const uchar* p = &src[(y * w + x) * ch];
        return (p[0] * 77 + p[1] * 150 + p[2] * 29) >> 8;
    };
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < w * h; i++) {
        dst[i * ch + 0] = 0;
        dst[i * ch + 1] = 0;
        dst[i * ch + 2] = 0;
        if (ch == 4) dst[i * ch + 3] = 255;
    }
    
    #pragma omp parallel for schedule(static)
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            int L00 = get_luminance(x - 1, y - 1);
            int L01 = get_luminance(x, y - 1);
            int L02 = get_luminance(x + 1, y - 1);

            int L10 = get_luminance(x - 1, y);
            int L12 = get_luminance(x + 1, y);

            int L20 = get_luminance(x - 1, y + 1);
            int L21 = get_luminance(x, y + 1);
            int L22 = get_luminance(x + 1, y + 1);

            int Gx = -L00 + L02 - 2 * L10 + 2 * L12 - L20 + L22;
            int Gy = -L00 - 2 * L01 - L02 + L20 + 2 * L21 + L22;

            int val = std::abs(Gx) + std::abs(Gy);
            if (val > 255) val = 255;

            int idx = (y * w + x) * ch;
            dst[idx + 0] = val;
            dst[idx + 1] = val;
            dst[idx + 2] = val;
        }
    }
}

void edges_simd(const uchar* src, uchar* dst, int w, int h, int ch) {
    std::vector<uchar> L(w * h + 16, 0);
    for (int i = 0; i < w * h; i++) {
        const uchar* p = &src[i * ch];
        L[i] = (uchar)((p[0] * 77 + p[1] * 150 + p[2] * 29) >> 8);
    }

    for (int i = 0; i < w * h; i++) {
        dst[i * ch] = 0;
        dst[i * ch + 1] = 0;
        dst[i * ch + 2] = 0;
        if (ch == 4) dst[i * ch + 3] = 255;
    }

    const __m256i v255 = _mm256_set1_epi16(255);

    for (int y = 1; y < h - 1; y++) {
        int x = 1;
        for (; x + 16 <= w - 1; x += 16) {
            int c = y * w + x;

            auto ld = [&](int off) -> __m256i {
                return _mm256_cvtepu8_epi16(
                    _mm_loadu_si128((const __m128i*)(L.data() + c + off)));
            };

            __m256i tl = ld(-w - 1), tc = ld(-w), tr = ld(-w + 1);
            __m256i ml = ld(-1), mr = ld(1);
            __m256i bl = ld(w - 1), bc = ld(w), br = ld(w + 1);

            __m256i gx = _mm256_add_epi16(
                _mm256_sub_epi16(tr, tl),
                _mm256_add_epi16(_mm256_slli_epi16(_mm256_sub_epi16(mr, ml), 1),
                    _mm256_sub_epi16(br, bl)));

            __m256i gy = _mm256_add_epi16(
                _mm256_sub_epi16(bl, tl),
                _mm256_add_epi16(_mm256_slli_epi16(_mm256_sub_epi16(bc, tc), 1),
                    _mm256_sub_epi16(br, tr)));

            __m256i mag = _mm256_add_epi16(_mm256_abs_epi16(gx), _mm256_abs_epi16(gy));
            mag = _mm256_min_epu16(mag, v255);

            alignas(32) uint16_t tmp[16];
            _mm256_store_si256((__m256i*)tmp, mag);
            for (int j = 0; j < 16; j++) {
                int idx = (c + j) * ch;
                uchar e = (uchar)tmp[j];
                dst[idx] = e; dst[idx + 1] = e; dst[idx + 2] = e;
            }
        }

        for (; x < w - 1; x++) {
            auto lum = [&](int xx, int yy) -> int {
                const uchar* p = &src[(yy * w + xx) * ch];
                return (p[0] * 77 + p[1] * 150 + p[2] * 29) >> 8;
                };

            int L00 = lum(x - 1, y - 1), L01 = lum(x, y - 1), L02 = lum(x + 1, y - 1);
            int L10 = lum(x - 1, y), L12 = lum(x + 1, y);
            int L20 = lum(x - 1, y + 1), L21 = lum(x, y + 1), L22 = lum(x + 1, y + 1);

            int Gx = -L00 + L02 - 2 * L10 + 2 * L12 - L20 + L22;
            int Gy = -L00 - 2 * L01 - L02 + L20 + 2 * L21 + L22;

            int val = std::abs(Gx) + std::abs(Gy);
            if (val > 255) val = 255;

            int idx = (y * w + x) * ch;
            dst[idx] = val; dst[idx + 1] = val; dst[idx + 2] = val;
        }
    }
}

void edges_ocl(const uchar* src, uchar* dst, int w, int h, int ch, OCLCtx& ctx) {
    if (!ctx.isValid || !ctx.k_edges) { std::memcpy(dst, src, (size_t)w * h * ch); return; }
    run_kernel(ctx.k_edges, src, dst, w, h, ch, ctx);
}


int main() {
    setlocale(LC_ALL, "Russian");

    const char* imgPath = "input2400.png";

    OCLCtx ctx = init_opencl();
    build_ocl_kernels(ctx);

    int width = 0, height = 0, channels = 0;
    uchar* src = stbi_load(imgPath, &width, &height, &channels, 0);
    if (!src) { release_opencl(ctx); return 1; }

    uchar* dst = new uchar[width * height * channels];

    invert_seq(src, dst, width, height, channels);

    struct Row { const char* name; double s, o, si, cl; };
    Row rows[3];

    rows[0] = { "Инверсия",
        measure_ms([&] {invert_seq(src, dst, width, height, channels); }, 5),
        measure_ms([&] {invert_omp(src, dst, width, height, channels); }, 5),
        measure_ms([&] {invert_simd(src, dst, width, height, channels); }, 5),
        measure_ms([&] {invert_ocl(src, dst, width, height, channels, ctx); }, 5) };

    rows[1] = { "Медиана",
        measure_ms([&] {median_seq(src, dst, width, height, channels); }, 5),
        measure_ms([&] {median_omp(src, dst, width, height, channels); }, 5),
        measure_ms([&] {median_simd(src, dst, width, height, channels); }, 5),
        measure_ms([&] {median_ocl(src, dst, width, height, channels, ctx); }, 5) };

    rows[2] = { "Границы",
        measure_ms([&] {edges_seq(src, dst, width, height, channels); }, 5),
        measure_ms([&] {edges_omp(src, dst, width, height, channels); }, 5),
        measure_ms([&] {edges_simd(src, dst, width, height, channels); }, 5),
        measure_ms([&] {edges_ocl(src, dst, width, height, channels, ctx); }, 5) };

    const int W1 = 12, W2 = 10, W3 = 10, W4 = 10, W5 = 9;

    std::cout << std::left
        << std::setw(W1) << "Фильтр"
        << std::setw(W2) << "seq, мс"
        << std::setw(W3) << "omp, мс"
        << std::setw(W4) << "simd, мс"
        << std::setw(W5) << "ocl, мс"
        << std::setw(W5) << "х omp"
        << std::setw(W5) << "х simd"
        << std::setw(W5) << "х ocl" << "\n";

    std::cout << std::fixed << std::setprecision(3);
    for (auto& r : rows) {
        std::cout << std::left
            << std::setw(W1) << r.name
            << std::setw(W2) << r.s
            << std::setw(W3) << r.o
            << std::setw(W4) << r.si
            << std::setw(W5) << r.cl
            << std::setprecision(2)
            << std::setw(W5) << r.s / r.o
            << std::setw(W5) << r.s / r.si
            << std::setw(W5) << r.s / r.cl
            << "\n";
        std::cout << std::setprecision(3);
    }

    median_seq(src, dst, width, height, channels);
    stbi_write_png("output_median.png", width, height, channels, dst, width * channels);

    invert_seq(src, dst, width, height, channels);
    stbi_write_png("output_invert.png", width, height, channels, dst, width * channels);

    edges_seq(src, dst, width, height, channels);
    stbi_write_png("output_edges.png", width, height, channels, dst, width * channels);

    stbi_image_free(src);
    delete[] dst;
    release_opencl(ctx);
    return 0;
}