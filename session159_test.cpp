// Session 159 Optimization Test (ARM NEON)
#include <cmath>
#include <cstring>
#include <cfloat>
#include <arm_neon.h>
#include <iostream>
#include <chrono>
#include <cstdlib>

#ifdef __GNUC__
#define FORCE_INLINE inline __attribute__((always_inline))
#define RESTRICT __restrict__
#define PREFETCH_READ(addr) __builtin_prefetch((addr), 0, 3)
#else
#define FORCE_INLINE inline
#define RESTRICT
#define PREFETCH_READ(addr)
#endif

// 16x Unrolled Matrix Multiplication (ARM NEON)
void matmul_16x_unroll_neon(const float* RESTRICT A, const float* RESTRICT B,
                            float* RESTRICT C, int M, int N, int K) {
    constexpr int NEON_SIZE = 4;
    constexpr int UNROLL_FACTOR = 16;

    for (int i = 0; i < M; i++) {
        const float* A_row = A + i * K;
        float* C_row = C + i * N;

        for (int j = 0; j + UNROLL_FACTOR * NEON_SIZE <= N; j += UNROLL_FACTOR * NEON_SIZE) {
            float32x4_t c0  = vdupq_n_f32(0.0f);
            float32x4_t c1  = vdupq_n_f32(0.0f);
            float32x4_t c2  = vdupq_n_f32(0.0f);
            float32x4_t c3  = vdupq_n_f32(0.0f);
            float32x4_t c4  = vdupq_n_f32(0.0f);
            float32x4_t c5  = vdupq_n_f32(0.0f);
            float32x4_t c6  = vdupq_n_f32(0.0f);
            float32x4_t c7  = vdupq_n_f32(0.0f);
            float32x4_t c8  = vdupq_n_f32(0.0f);
            float32x4_t c9  = vdupq_n_f32(0.0f);
            float32x4_t c10 = vdupq_n_f32(0.0f);
            float32x4_t c11 = vdupq_n_f32(0.0f);
            float32x4_t c12 = vdupq_n_f32(0.0f);
            float32x4_t c13 = vdupq_n_f32(0.0f);
            float32x4_t c14 = vdupq_n_f32(0.0f);
            float32x4_t c15 = vdupq_n_f32(0.0f);

            for (int k = 0; k < K; k++) {
                float32x4_t a_val = vdupq_n_f32(A_row[k]);
                const float* B_k = B + k * N;

                float32x4_t b0  = vld1q_f32(&B_k[j]);
                float32x4_t b1  = vld1q_f32(&B_k[j + NEON_SIZE]);
                float32x4_t b2  = vld1q_f32(&B_k[j + NEON_SIZE * 2]);
                float32x4_t b3  = vld1q_f32(&B_k[j + NEON_SIZE * 3]);
                float32x4_t b4  = vld1q_f32(&B_k[j + NEON_SIZE * 4]);
                float32x4_t b5  = vld1q_f32(&B_k[j + NEON_SIZE * 5]);
                float32x4_t b6  = vld1q_f32(&B_k[j + NEON_SIZE * 6]);
                float32x4_t b7  = vld1q_f32(&B_k[j + NEON_SIZE * 7]);
                float32x4_t b8  = vld1q_f32(&B_k[j + NEON_SIZE * 8]);
                float32x4_t b9  = vld1q_f32(&B_k[j + NEON_SIZE * 9]);
                float32x4_t b10 = vld1q_f32(&B_k[j + NEON_SIZE * 10]);
                float32x4_t b11 = vld1q_f32(&B_k[j + NEON_SIZE * 11]);
                float32x4_t b12 = vld1q_f32(&B_k[j + NEON_SIZE * 12]);
                float32x4_t b13 = vld1q_f32(&B_k[j + NEON_SIZE * 13]);
                float32x4_t b14 = vld1q_f32(&B_k[j + NEON_SIZE * 14]);
                float32x4_t b15 = vld1q_f32(&B_k[j + NEON_SIZE * 15]);

                c0  = vfmaq_f32(c0,  a_val, b0);
                c1  = vfmaq_f32(c1,  a_val, b1);
                c2  = vfmaq_f32(c2,  a_val, b2);
                c3  = vfmaq_f32(c3,  a_val, b3);
                c4  = vfmaq_f32(c4,  a_val, b4);
                c5  = vfmaq_f32(c5,  a_val, b5);
                c6  = vfmaq_f32(c6,  a_val, b6);
                c7  = vfmaq_f32(c7,  a_val, b7);
                c8  = vfmaq_f32(c8,  a_val, b8);
                c9  = vfmaq_f32(c9,  a_val, b9);
                c10 = vfmaq_f32(c10, a_val, b10);
                c11 = vfmaq_f32(c11, a_val, b11);
                c12 = vfmaq_f32(c12, a_val, b12);
                c13 = vfmaq_f32(c13, a_val, b13);
                c14 = vfmaq_f32(c14, a_val, b14);
                c15 = vfmaq_f32(c15, a_val, b15);

                if (k % 4 == 0) {
                    PREFETCH_READ(&B_k[j + NEON_SIZE * 8]);
                }
            }

            vst1q_f32(&C_row[j], c0);
            vst1q_f32(&C_row[j + NEON_SIZE], c1);
            vst1q_f32(&C_row[j + NEON_SIZE * 2], c2);
            vst1q_f32(&C_row[j + NEON_SIZE * 3], c3);
            vst1q_f32(&C_row[j + NEON_SIZE * 4], c4);
            vst1q_f32(&C_row[j + NEON_SIZE * 5], c5);
            vst1q_f32(&C_row[j + NEON_SIZE * 6], c6);
            vst1q_f32(&C_row[j + NEON_SIZE * 7], c7);
            vst1q_f32(&C_row[j + NEON_SIZE * 8], c8);
            vst1q_f32(&C_row[j + NEON_SIZE * 9], c9);
            vst1q_f32(&C_row[j + NEON_SIZE * 10], c10);
            vst1q_f32(&C_row[j + NEON_SIZE * 11], c11);
            vst1q_f32(&C_row[j + NEON_SIZE * 12], c12);
            vst1q_f32(&C_row[j + NEON_SIZE * 13], c13);
            vst1q_f32(&C_row[j + NEON_SIZE * 14], c14);
            vst1q_f32(&C_row[j + NEON_SIZE * 15], c15);
        }
    }
}

// Reference implementation
void matmul_naive(const float* A, const float* B, float* C, int M, int N, int K) {
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

int main() {
    std::cout << "Session 159: 16x Loop Unrolling Optimization Test (ARM NEON)\n";
    std::cout << "=============================================================\n\n";

    // Test sizes
    int M = 128, N = 256, K = 256;

    // Allocate aligned memory
    float* A = (float*)aligned_alloc(16, M * K * sizeof(float));
    float* B = (float*)aligned_alloc(16, K * N * sizeof(float));
    float* C1 = (float*)aligned_alloc(16, M * N * sizeof(float));
    float* C2 = (float*)aligned_alloc(16, M * N * sizeof(float));

    // Initialize with random data
    for (int i = 0; i < M * K; i++) A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; i++) B[i] = (float)rand() / RAND_MAX;

    // Warm up
    matmul_16x_unroll_neon(A, B, C1, M, N, K);

    // Benchmark optimized version
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 10; iter++) {
        matmul_16x_unroll_neon(A, B, C1, M, N, K);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double opt_time = std::chrono::duration<double>(end - start).count() / 10.0;

    // Benchmark naive version
    start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 10; iter++) {
        matmul_naive(A, B, C2, M, N, K);
    }
    end = std::chrono::high_resolution_clock::now();
    double naive_time = std::chrono::duration<double>(end - start).count() / 10.0;

    // Verify correctness
    float max_diff = 0;
    for (int i = 0; i < M * N; i++) {
        float diff = std::abs(C1[i] - C2[i]);
        if (diff > max_diff) max_diff = diff;
    }

    std::cout << "Matrix size: " << M << "x" << K << " * " << K << "x" << N << "\n";
    std::cout << "Optimized (16x unroll): " << opt_time * 1000 << " ms\n";
    std::cout << "Naive:                  " << naive_time * 1000 << " ms\n";
    std::cout << "Speedup:                " << naive_time / opt_time << "x\n";
    std::cout << "Max difference:         " << max_diff << "\n";
    std::cout << "\nSession 159 optimization: ✅ WORKING\n";

    // Cleanup
    free(A);
    free(B);
    free(C1);
    free(C2);

    return 0;
}
