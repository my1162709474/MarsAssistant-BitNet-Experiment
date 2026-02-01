/**
 * BitNet Optimized Test - Cross-Platform (ARM NEON + x86 AVX2)
 * Simplified, compilable version
 */

#include <cmath>
#include <cstring>
#include <cfloat>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <vector>

// Platform detection
#if defined(__x86_64__) || defined(__i386__)
#define IS_X86_PLATFORM 1
#include <immintrin.h>
#else
#define IS_X86_PLATFORM 0
#endif

#if defined(__aarch64__) || defined(__arm__)
#include <arm_neon.h>
#endif

#include <pthread.h>

// Configuration
constexpr int BLOCK_SIZE = 64;
constexpr int CACHE_LINE_SIZE = 64;

// Compiler hints
#ifdef __GNUC__
#define HOT_FUNC __attribute__((hot))
#define ALIGNED __attribute__((aligned(32)))
#define RESTRICT __restrict__
#else
#define HOT_FUNC
#define ALIGNED
#define RESTRICT
#endif

// ==================== Matrix Multiplication ====================

#if IS_X86_PLATFORM

// AVX2 implementation
void matmul_avx2(const float* A, const float* B, float* C,
                 int M, int N, int K) {
    constexpr int AVX_SIZE = 8;
    
    for (int i = 0; i < M; i++) {
        const float* A_row = A + i * K;
        float* C_row = C + i * N;
        
        // Use dynamic allocation for large matrices
        int num_vec = N / AVX_SIZE;
        std::vector<__m256> c_vec(num_vec);
        for (int j = 0; j < num_vec; j++) {
            c_vec[j] = _mm256_setzero_ps();
        }
        
        for (int k = 0; k < K; k++) {
            __m256 a_val = _mm256_set1_ps(A_row[k]);
            const float* B_k = B + k * N;
            
            for (int j = 0; j < num_vec; j++) {
                __m256 b_vec = _mm256_loadu_ps(&B_k[j * AVX_SIZE]);
                c_vec[j] = _mm256_fmadd_ps(a_val, b_vec, c_vec[j]);
            }
        }
        
        for (int j = 0; j < num_vec; j++) {
            _mm256_storeu_ps(&C_row[j * AVX_SIZE], c_vec[j]);
        }
    }
}

#else

// ARM NEON implementation
void matmul_neon(const float* A, const float* B, float* C,
                 int M, int N, int K) {
    constexpr int NEON_SIZE = 4;
    
    for (int i = 0; i < M; i++) {
        const float* A_row = A + i * K;
        float* C_row = C + i * N;
        
        // Use dynamic allocation for large matrices
        int num_vec = N / NEON_SIZE;
        std::vector<float32x4_t> c_vec(num_vec);
        for (int j = 0; j < num_vec; j++) {
            c_vec[j] = vdupq_n_f32(0.0f);
        }
        
        for (int k = 0; k < K; k++) {
            float32x4_t a_val = vdupq_n_f32(A_row[k]);
            const float* B_k = B + k * N;
            
            for (int j = 0; j < num_vec; j++) {
                float32x4_t b_vec = vld1q_f32(&B_k[j * NEON_SIZE]);
                c_vec[j] = vfmaq_f32(c_vec[j], a_val, b_vec);
            }
        }
        
        for (int j = 0; j < num_vec; j++) {
            vst1q_f32(&C_row[j * NEON_SIZE], c_vec[j]);
        }
    }
}

#endif

// Naive fallback
void matmul_naive(const float* A, const float* B, float* C, 
                  int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// ==================== ReLU Activation ====================

void relu(float* data, int size) {
#if IS_X86_PLATFORM
    constexpr int AVX_SIZE = 8;
    __m256 zero = _mm256_setzero_ps();
    
    for (int i = 0; i + AVX_SIZE <= size; i += AVX_SIZE) {
        __m256 vals = _mm256_loadu_ps(&data[i]);
        vals = _mm256_max_ps(vals, zero);
        _mm256_storeu_ps(&data[i], vals);
    }
    for (int i = size - (size % AVX_SIZE); i < size; i++) {
        data[i] = std::max(0.0f, data[i]);
    }
#else
    constexpr int NEON_SIZE = 4;
    float32x4_t zero = vdupq_n_f32(0.0f);
    
    for (int i = 0; i + NEON_SIZE <= size; i += NEON_SIZE) {
        float32x4_t vals = vld1q_f32(&data[i]);
        vals = vmaxq_f32(vals, zero);
        vst1q_f32(&data[i], vals);
    }
    for (int i = size - (size % NEON_SIZE); i < size; i++) {
        data[i] = std::max(0.0f, data[i]);
    }
#endif
}

// ==================== Benchmark ====================

void benchmark(const std::string& name, 
               void (*func)(const float*, const float*, float*, int, int, int),
               const float* A, const float* B, float* C,
               int M, int N, int K, int iterations = 100) {
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        func(A, B, C, M, N, K);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double avg_time = duration.count() / (double)iterations;
    double gflops = (2.0 * M * N * K) / (avg_time * 1000.0);
    
    std::cout << name << ": " << avg_time << " us, " << gflops << " GFLOPS" << std::endl;
}

// ==================== Main ====================

int main() {
    std::cout << "BitNet Performance Optimization Test" << std::endl;
#if IS_X86_PLATFORM
    std::cout << "Platform: x86_64 (AVX2)" << std::endl;
#else
    std::cout << "Platform: ARM (NEON)" << std::endl;
#endif
    std::cout << std::endl;
    
    // Test sizes
    const int M = 512, N = 512, K = 512;
    const int iterations = 50;
    
    // Allocate aligned memory
    float* A = (float*)aligned_alloc(CACHE_LINE_SIZE, sizeof(float) * M * K);
    float* B = (float*)aligned_alloc(CACHE_LINE_SIZE, sizeof(float) * K * N);
    float* C = (float*)aligned_alloc(CACHE_LINE_SIZE, sizeof(float) * M * N);
    
    // Initialize with random data
    std::srand(42);
    for (int i = 0; i < M * K; i++) A[i] = (float)std::rand() / RAND_MAX - 0.5f;
    for (int i = 0; i < K * N; i++) B[i] = (float)std::rand() / RAND_MAX - 0.5f;
    
    std::cout << "Matrix size: " << M << "x" << N << "x" << K << std::endl;
    std::cout << "Iterations: " << iterations << std::endl << std::endl;
    
    // Run benchmarks
    std::cout << "=== Benchmarks ===" << std::endl;
    benchmark("Naive", matmul_naive, A, B, C, M, N, K, iterations);
    
#if IS_X86_PLATFORM
    benchmark("AVX2", matmul_avx2, A, B, C, M, N, K, iterations);
#else
    benchmark("NEON", matmul_neon, A, B, C, M, N, K, iterations);
#endif
    
    // Cleanup
    free(A);
    free(B);
    free(C);
    
    return 0;
}
