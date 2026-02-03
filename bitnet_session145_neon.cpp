// Session 145 Performance Optimization Test (ARM NEON)
// Focus: Ultra-Extreme Optimizations for Apple Silicon

#include <cmath>
#include <cstring>
#include <cfloat>
#include <arm_neon.h>
#include <pthread.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <thread>
#include <atomic>
#include <mutex>
#include <random>

// Compiler hints
#ifdef __GNUC__
#define HOT_FUNC __attribute__((hot))
#define ALIGNED __attribute__((aligned(32)))
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#define RESTRICT __restrict__
#define FORCE_INLINE inline __attribute__((always_inline))
#define PREFETCH_READ(addr) __builtin_prefetch((addr), 0, 3)
#else
#define FORCE_INLINE inline
#define PREFETCH_READ(addr)
#define RESTRICT
#endif

// Global counters
static std::atomic<size_t> session145_ops{0};
static std::atomic<size_t> session145_matmul_ops{0};

// ==================== Ultra 256x Loop Unrolling (NEON) ====================

FORCE_INLINE void matmul_256x_ultra_unroll_neon(const float* RESTRICT A,
                                                  const float* RESTRICT B,
                                                  float* RESTRICT C,
                                                  int M, int N, int K) {
    constexpr int NEON_SIZE = 4;
    constexpr int UNROLL_N = 64;
    constexpr int UNROLL_K = 16;

    for (int i = 0; i < M; i++) {
        const float* A_row = A + i * K;

        for (int j = 0; j + UNROLL_N <= N; j += UNROLL_N) {
            float32x4_t acc[UNROLL_N / NEON_SIZE];
            for (int aj = 0; aj < UNROLL_N / NEON_SIZE; aj++) {
                acc[aj] = vdupq_n_f32(0.0f);
            }

            for (int kk = 0; kk < K; kk += UNROLL_K) {
                int k_end = std::min(kk + UNROLL_K, K);

                for (int k = kk; k < k_end; k++) {
                    PREFETCH_READ(A_row + k + 16);
                    float32x4_t a_val = vdupq_n_f32(A_row[k]);

                    for (int bj = 0; bj < UNROLL_N; bj += NEON_SIZE) {
                        float32x4_t b_vec = vld1q_f32(B + k * N + j + bj);
                        acc[bj / NEON_SIZE] = vfmaq_f32(acc[bj / NEON_SIZE], a_val, b_vec);
                    }
                }
            }

            for (int bj = 0; bj < UNROLL_N; bj += NEON_SIZE) {
                float32x4_t c_vec = vld1q_f32(C + i * N + j + bj);
                vst1q_f32(C + i * N + j + bj, vaddq_f32(c_vec, acc[bj / NEON_SIZE]));
            }
        }

        for (int j = (N / UNROLL_N) * UNROLL_N; j < N; j += NEON_SIZE) {
            int remaining = std::min(NEON_SIZE, N - j);
            float32x4_t acc = vdupq_n_f32(0.0f);

            for (int k = 0; k < K; k++) {
                float32x4_t a_val = vdupq_n_f32(A_row[k]);
                float32x4_t b_vec = vld1q_f32(B + k * N + j);
                acc = vfmaq_f32(acc, a_val, b_vec);
            }

            float32x4_t c_vec = vld1q_f32(C + i * N + j);
            vst1q_f32(C + i * N + j, vaddq_f32(c_vec, acc));
        }
    }

    session145_matmul_ops.fetch_add(M * N * K);
}

FORCE_INLINE void matmul_neon(const float* A, const float* B, float* C, int M, int N, int K) {
    constexpr int NEON_SIZE = 4;

    for (int i = 0; i < M; i++) {
        const float* A_row = A + i * K;

        for (int j = 0; j < N; j += NEON_SIZE) {
            float32x4_t acc = vdupq_n_f32(0.0f);

            for (int k = 0; k < K; k++) {
                float32x4_t a_val = vdupq_n_f32(A_row[k]);
                float32x4_t b_vec = vld1q_f32(B + k * N + j);
                acc = vfmaq_f32(acc, a_val, b_vec);
            }

            float32x4_t c_vec = vld1q_f32(C + i * N + j);
            vst1q_f32(C + i * N + j, vaddq_f32(c_vec, acc));
        }
    }
}

// ==================== Aggressive Prefetch + Cache Blocking (NEON) ====================

FORCE_INLINE void matmul_cache_blocking_aggressive_neon(const float* RESTRICT A,
                                                         const float* RESTRICT B,
                                                         float* RESTRICT C,
                                                         int M, int N, int K) {
    constexpr int BLOCK_M = 64;
    constexpr int BLOCK_N = 64;
    constexpr int BLOCK_K = 32;
    constexpr int NEON_SIZE = 4;
    constexpr int PREFETCH_DIST = 32;

    for (int mb = 0; mb < M; mb += BLOCK_M) {
        int mb_end = std::min(mb + BLOCK_M, M);

        for (int nb = 0; nb < N; nb += BLOCK_N) {
            int nb_end = std::min(nb + BLOCK_N, N);

            for (int kb = 0; kb < K; kb += BLOCK_K) {
                int kb_end = std::min(kb + BLOCK_K, K);

                for (int k = kb; k < kb_end; k++) {
                    for (int n = nb; n < nb_end && n < nb + 8; n += NEON_SIZE) {
                        PREFETCH_READ(B + k * N + n + PREFETCH_DIST);
                    }
                }

                for (int i = mb; i < mb_end; i++) {
                    const float* A_row = A + i * K;

                    for (int j = nb; j < nb_end; j += NEON_SIZE) {
                        float32x4_t acc = vdupq_n_f32(0.0f);

                        for (int k = kb; k < kb_end; k++) {
                            float32x4_t a_val = vdupq_n_f32(A_row[k]);
                            float32x4_t b_vec = vld1q_f32(B + k * N + j);
                            acc = vfmaq_f32(acc, a_val, b_vec);
                        }

                        float32x4_t c_vec = vld1q_f32(C + i * N + j);
                        vst1q_f32(C + i * N + j, vaddq_f32(c_vec, acc));
                    }
                }
            }
        }
    }
}

// ==================== Hyper-Parallel Matrix Multiply (NEON) ====================

FORCE_INLINE void matmul_hyper_parallel_neon(const float* A,
                                              const float* B,
                                              float* C,
                                              int M, int N, int K) {
    constexpr int CHUNK_SIZE = 32;
    const int NUM_THREADS = std::thread::hardware_concurrency();

    std::vector<std::thread> threads;
    std::atomic<int> next_row{0};

    auto worker = [&](int thread_id) {
        int local_ops = 0;

        while (true) {
            int start_row = next_row.fetch_add(CHUNK_SIZE, std::memory_order_relaxed);
            if (start_row >= M) break;

            int end_row = std::min(start_row + CHUNK_SIZE, M);

            for (int i = start_row; i < end_row; i++) {
                const float* A_row = A + i * K;
                for (int j = 0; j < N; j += 4) {
                    float32x4_t acc = vdupq_n_f32(0.0f);
                    for (int k = 0; k < K; k++) {
                        float32x4_t a_val = vdupq_n_f32(A_row[k]);
                        float32x4_t b_vec = vld1q_f32(B + k * N + j);
                        acc = vfmaq_f32(acc, a_val, b_vec);
                    }
                    float32x4_t c_vec = vld1q_f32(C + i * N + j);
                    vst1q_f32(C + i * N + j, vaddq_f32(c_vec, acc));
                }
                local_ops += N * K;
            }
        }

        session145_ops.fetch_add(local_ops);
    };

    for (int t = 0; t < NUM_THREADS; t++) {
        threads.emplace_back(worker, t);
    }

    for (auto& t : threads) {
        t.join();
    }
}

int main() {
    std::cout << "BitNet Session 145 Performance Test (ARM NEON)" << std::endl;
    std::cout << "================================================" << std::endl;

    int M = 512, N = 512, K = 512;
    int iterations = 100;

    std::cout << "Matrix size: " << M << "x" << N << "x" << K << std::endl;
    std::cout << "Iterations: " << iterations << std::endl;

    // Allocate aligned memory
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    posix_memalign((void**)&A, 64, M * K * sizeof(float));
    posix_memalign((void**)&B, 64, K * N * sizeof(float));
    posix_memalign((void**)&C, 64, M * N * sizeof(float));

    // Initialize
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < M * K; i++) A[i] = dist(rng);
    for (int i = 0; i < K * N; i++) B[i] = dist(rng);
    for (int i = 0; i < M * N; i++) C[i] = 0.0f;

    // Test methods
    std::vector<std::pair<std::string, std::function<void()>>> methods = {
        {"matmul_neon", [&]() {
            std::fill(C, C + M * N, 0.0f);
            matmul_neon(A, B, C, M, N, K);
        }},
        {"matmul_256x_ultra_unroll_neon", [&]() {
            std::fill(C, C + M * N, 0.0f);
            matmul_256x_ultra_unroll_neon(A, B, C, M, N, K);
        }},
        {"matmul_cache_blocking_aggressive_neon", [&]() {
            std::fill(C, C + M * N, 0.0f);
            matmul_cache_blocking_aggressive_neon(A, B, C, M, N, K);
        }},
        {"matmul_hyper_parallel_neon", [&]() {
            std::fill(C, C + M * N, 0.0f);
            matmul_hyper_parallel_neon(A, B, C, M, N, K);
        }},
    };

    for (const auto& [name, func] : methods) {
        auto start = std::chrono::high_resolution_clock::now();

        for (int iter = 0; iter < iterations; iter++) {
            func();
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        double avg_time = static_cast<double>(duration) / iterations;
        double gflops = 2.0 * M * N * K / (avg_time * 1000.0);

        std::cout << name << ": " << avg_time << " us (" << gflops << " GFLOPS)" << std::endl;
    }

    free(A);
    free(B);
    free(C);

    return 0;
}
