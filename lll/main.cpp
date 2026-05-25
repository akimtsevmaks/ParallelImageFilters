#define _CRT_SECURE_NO_WARNINGS
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <iostream>
#include <chrono>
#include <string>
#include <vector>
#include <algorithm>
#include <functional>
#include <cmath>
#include <immintrin.h>
#include <omp.h>
#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>
#include <cstring>
#include <iomanip>

using uchar = unsigned char;


struct OCLCtx {
	cl_platform_id platform = nullptr;
	cl_device_id device = nullptr;
	cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    bool isValid = false;
};


OCLCtx init_opencl() {
	OCLCtx ctx;
	cl_uint numPlatforms = 0;
	cl_int err = clGetPlatformIDs(0, nullptr, &numPlatforms);

    if (err != CL_SUCCESS || numPlatforms == 0) {
		std::cout << "Ошибка получения платформ OpenCL" << std::endl;
		return ctx;
    }

	std::vector<cl_platform_id> platforms(numPlatforms);
	clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);

	cl_device_id selectedDevice = nullptr; 
	cl_platform_id selectedPlatform = nullptr;
	bool deviceFound = false;

    auto findDevice = [&](cl_device_type type, bool forceIntel) -> bool {
        for (auto p : platforms) {
            if (forceIntel) {
                char vendor[256] = { 0 };
                clGetPlatformInfo(p, CL_PLATFORM_NAME, sizeof(vendor), vendor, nullptr);
                if (std::string(vendor).find("Intel") == std::string::npos) {
                    continue;
                }
            }

            cl_uint numDevices = 0;
            cl_uint devErr = clGetDeviceIDs(p, type, 0, nullptr, &numDevices);

            if (devErr == CL_SUCCESS && numDevices > 0) {
                std::vector<cl_device_id> devices(numDevices);
                clGetDeviceIDs(p, type, numDevices, devices.data(), nullptr);
                selectedDevice = devices[0];
                selectedPlatform = p;
                return true;
            }
        }
        return false;
    };

    if (findDevice(CL_DEVICE_TYPE_GPU, true)) {
        deviceFound = true;
    }
	else if (findDevice(CL_DEVICE_TYPE_CPU, true)) {
		deviceFound = true;
	}
	else if (findDevice(CL_DEVICE_TYPE_GPU, false)) {
		deviceFound = true;
	}
	else if (findDevice(CL_DEVICE_TYPE_CPU, false)) {
		deviceFound = true;
	}

	if (!deviceFound) {
		std::cout << "Не найдено подходящее устройство OpenCL" << std::endl;
		return ctx;
	}

	ctx.platform = selectedPlatform;
	ctx.device = selectedDevice;

	ctx.context = clCreateContext(nullptr, 1, &ctx.device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
		std::cout << "Ошибка создания контекста OpenCL" << std::endl;
		return ctx;
    }

	ctx.queue = clCreateCommandQueueWithProperties(ctx.context, ctx.device, 0, &err);
    if (err != CL_SUCCESS) {
		ctx.queue = clCreateCommandQueue(ctx.context, ctx.device, 0, &err);
    }

    if (err != CL_SUCCESS) {
		std::cout << "Ошибка создания очереди OpenCL" << std::endl;
		clReleaseContext(ctx.context);
        ctx.context = nullptr;
        return ctx;
    }

	ctx.isValid = true;

	char deviceName[256] = { 0 };
	clGetDeviceInfo(ctx.device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);

	cl_ulong localMemSize = 0;
	clGetDeviceInfo(ctx.device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(localMemSize), &localMemSize, nullptr);

	size_t maxWorkGroupSize = 0;
	clGetDeviceInfo(ctx.device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkGroupSize), &maxWorkGroupSize, nullptr);

	std::cout << "Выбранное устройство: " << deviceName << std::endl;
	std::cout << "Локальная память: " << localMemSize / 1024 << " KB" << std::endl;
	std::cout << "Макс. размер рабочей группы: " << maxWorkGroupSize << std::endl;

    return ctx;
}

double measure_ms(std::function<void()> fn, int runs = 5) {
    if (runs <= 0) return 0.0;

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
    std::memcpy(dst, src, w * h * ch);
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
    std::memcpy(dst, src, w * h * ch);
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
        dst[i * ch] = 0; dst[i * ch + 1] = 0; dst[i * ch + 2] = 0;
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
            __m256i ml = ld(-1), mr = ld(+1);
            __m256i bl = ld(+w - 1), bc = ld(+w), br = ld(+w + 1);

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
            int val = std::abs(Gx) + std::abs(Gy); if (val > 255) val = 255;
            int idx = (y * w + x) * ch;
            dst[idx] = val; dst[idx + 1] = val; dst[idx + 2] = val;
        }
    }
}

void edges_ocl(const uchar* src, uchar* dst, int w, int h, int ch, OCLCtx& ctx) {
    std::memcpy(dst, src, w * h * ch);
}


int main() {
    setlocale(LC_ALL, "Russian");

    OCLCtx ctx = init_opencl();

    int width = 0, height = 0, channels = 0;
    unsigned char* data = stbi_load("input.png", &width, &height, &channels, 0);
    if (!data) {
        std::cout << "Ошибка загрузки изображения" << std::endl;
        if (ctx.isValid) {
            clReleaseCommandQueue(ctx.queue);
			clReleaseContext(ctx.context);
        }
        return 1;
    }


	size_t imgSize = static_cast<size_t>(width) * height * channels;
    uchar* dst = new uchar[imgSize];

    double t_med_seq = measure_ms([&]() { median_seq(data, dst, width, height, channels); });
    double t_med_omp = measure_ms([&]() { median_omp(data, dst, width, height, channels); });
    double t_med_simd = measure_ms([&]() { median_simd(data, dst, width, height, channels); });
    double t_med_ocl = measure_ms([&]() { median_ocl(data, dst, width, height, channels, ctx); });

    double t_inv_seq = measure_ms([&]() { invert_seq(data, dst, width, height, channels); });
    double t_inv_omp = measure_ms([&]() { invert_omp(data, dst, width, height, channels); });
    double t_inv_simd = measure_ms([&]() { invert_simd(data, dst, width, height, channels); });
    double t_inv_ocl = measure_ms([&]() { invert_ocl(data, dst, width, height, channels, ctx); });

    double t_edges_seq = measure_ms([&]() { edges_seq(data, dst, width, height, channels); });
    double t_edges_omp = measure_ms([&]() { edges_omp(data, dst, width, height, channels); });
    double t_edges_simd = measure_ms([&]() { edges_simd(data, dst, width, height, channels); });
    double t_edges_ocl = measure_ms([&]() { edges_ocl(data, dst, width, height, channels, ctx); });


    std::cout << std::left << std::setw(25) << "Фильтр"
        << " | " << std::setw(10) << "seq, мс"
        << " | " << std::setw(10) << "omp, мс"
        << " | " << std::setw(10) << "simd, мс"
        << " | " << std::setw(10) << "ocl, мс" << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << std::left << std::setw(25) << "Медианный фильтр"
        << " | " << std::setw(10) << t_med_seq
        << " | " << std::setw(10) << t_med_omp
        << " | " << std::setw(10) << t_med_simd
        << " | " << std::setw(10) << t_med_ocl << std::endl;

    std::cout << std::left << std::setw(25) << "Инверсия цвета"
        << " | " << std::setw(10) << t_inv_seq
        << " | " << std::setw(10) << t_inv_omp
        << " | " << std::setw(10) << t_inv_simd
        << " | " << std::setw(10) << t_inv_ocl << std::endl;

    std::cout << std::left << std::setw(25) << "Обнаружение границ"
        << " | " << std::setw(10) << t_edges_seq
        << " | " << std::setw(10) << t_edges_omp
        << " | " << std::setw(10) << t_edges_simd
        << " | " << std::setw(10) << t_edges_ocl << std::endl;

    std::cout << std::endl;


    median_seq(data, dst, width, height, channels);
    if (!stbi_write_png("output_median.png", width, height, channels, dst, width * channels)) {
        std::cout << "Ошибка сохранения output_median.png" << std::endl;
    }

    invert_seq(data, dst, width, height, channels);
    if (!stbi_write_png("output_invert.png", width, height, channels, dst, width * channels)) {
        std::cout << "Ошибка сохранения output_invert.png" << std::endl;
    }

    edges_seq(data, dst, width, height, channels);
    if (!stbi_write_png("output_edges.png", width, height, channels, dst, width * channels)) {
        std::cout << "Ошибка сохранения output_edges.png" << std::endl;
    }

    if (ctx.isValid) {
		clReleaseCommandQueue(ctx.queue);
		clReleaseContext(ctx.context);
    }

    stbi_image_free(data);
    delete[] dst;

    return 0;

}