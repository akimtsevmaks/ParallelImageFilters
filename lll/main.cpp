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
}

void median_simd(const uchar* src, uchar* dst, int w, int h, int ch) {
    std::memcpy(dst, src, w * h * ch);
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
    std::memcpy(dst, src, w * h * ch);
}

void invert_simd(const uchar* src, uchar* dst, int w, int h, int ch) {
    std::memcpy(dst, src, w * h * ch);
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
    std::memcpy(dst, src, w * h * ch);
}

void edges_simd(const uchar* src, uchar* dst, int w, int h, int ch) {
    std::memcpy(dst, src, w * h * ch);
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
    if (!stbi_write_png("output_median.png", width, height, channels, dst, 90)) {
        std::cout << "Ошибка сохранения output_median.png" << std::endl;
    }

    invert_seq(data, dst, width, height, channels);
    if (!stbi_write_png("output_invert.png", width, height, channels, dst, 90)) {
        std::cout << "Ошибка сохранения output_invert.png" << std::endl;
    }

    edges_seq(data, dst, width, height, channels);
    if (!stbi_write_png("output_edges.png", width, height, channels, dst, 90)) {
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