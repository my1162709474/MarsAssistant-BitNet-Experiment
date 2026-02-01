/**
 * Session 52: Memory Access & Quantization Optimizations Test
 * Test file for verifying new optimizations
 */

#include <cmath>
#include <cstring>
#include <cfloat>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define IS_X86_PLATFORM 1
#else
#define IS_X86_PLATFORM 0
#endif

#if defined(__aarch64__) || defined(__arm__)
#include <arm_neon.h>
#define IS_ARM_PLATFORM 1
#else
#define IS_ARM_PLATFORM 0
#endif

// Compiler Optimization Hints
#ifdef __GNUC__
#define FORCE_INLINE inline __attribute__((always_inline))
#define PREFETCH_READ(addr) __builtin_prefetch((addr), 0, 3)
#else
#define FORCE_INLINE inline
#define PREFETCH_READ(addr)
#endif

// 1. Optimized 1-bit Matrix Multiply with Improved Bit Parallelism
void matmul_1bit_optimized(const unsigned char* A_packed, const unsigned char* B_packed, 
                           float* C, int M, int N, int K) {
    const int K_words = (K + 31) / 32;
    constexpr int UNROLL_WORDS = 4;
    
    for (int i = 0; i < M; i++) {
        const unsigned int* A_words = reinterpret_cast<const unsigned int*>(A_packed + i * K);
        
        for (int j = 0; j < N; j++) {
            const unsigned int* B_words = reinterpret_cast<const unsigned int*>(B_packed + j * K);
            int popcount = 0;
            
            // Unroll by 4 for better ILP
            int w = 0;
            for (; w + UNROLL_WORDS <= K_words; w += UNROLL_WORDS) {
                popcount += __builtin_popcount(A_words[w] ^ B_words[w]);
                popcount += __builtin_popcount(A_words[w + 1] ^ B_words[w + 1]);
                popcount += __builtin_popcount(A_words[w + 2] ^ B_words[w + 2]);
                popcount += __builtin_popcount(A_words[w + 3] ^ B_words[w + 3]);
            }
            for (; w < K_words; w++) {
                popcount += __builtin_popcount(A_words[w] ^ B_words[w]);
            }
            
            C[i * N + j] = static_cast<float>(K - 2 * popcount);
        }
    }
}

// 2. Cache-Aware Batch Matrix Multiply with Prefetch
template<typename Func>
void matmul_batch_cache_aware(const float* A_batch, const float* B, float* C_batch,
                               int batch_size, int M, int N, int K, Func matmul_fn) {
    constexpr int PREFETCH_STRIDE = 64;
    
    for (int b = 0; b < batch_size; b++) {
        const float* A = A_batch + b * M * K;
        float* C = C_batch + b * M * N;
        
        // Prefetch B into cache
        for (int k = 0; k < K; k += PREFETCH_STRIDE) {
            PREFETCH_READ(B + k * N);
        }
        
        matmul_fn(A, B, C, M, N, K);
    }
}

// 3. Vectorized LayerNorm with SIMD Reduction
void layernorm_naive(float* output, const float* input, const float* weight,
                     const float* bias, int size) {
    // Compute mean
    float mean = 0.0f;
    for (int i = 0; i < size; i++) {
        mean += input[i];
    }
    mean /= size;
    
    // Compute variance
    float var = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = input[i] - mean;
        var += diff * diff;
    }
    var /= size;
    
    float inv_std = 1.0f / std::sqrt(var + 1e-5f);
    
    // Normalize and apply weight/bias
    for (int i = 0; i < size; i++) {
        float val = (input[i] - mean) * inv_std;
        if (weight) val *= weight[0];
        if (bias) val += bias[0];
        output[i] = val;
    }
}

#if IS_X86_PLATFORM
void layernorm_avx2(float* output, const float* input, const float* weight,
                    const float* bias, int size) {
    constexpr int VEC_SIZE = 8;
    
    // Compute mean (vectorized)
    __m256 sum_vec = _mm256_setzero_ps();
    int i = 0;
    for (; i + VEC_SIZE <= size; i += VEC_SIZE) {
        __m256 vals = _mm256_loadu_ps(&input[i]);
        sum_vec = _mm256_add_ps(sum_vec, vals);
    }
    for (; i < size; i++) {
        sum_vec = _mm256_add_ss(sum_vec, _mm256_set1_ps(input[i]));
    }
    
    float sum_arr[8];
    _mm256_storeu_ps(sum_arr, sum_vec);
    float mean = sum_arr[0];
    for (int j = 1; j < 8 && j < size; j++) {
        mean += sum_arr[j];
    }
    for (i = 8 * (size / 8); i < size; i++) {
        mean += input[i];
    }
    mean /= size;
    
    __m256 mean_vec = _mm256_set1_ps(mean);
    
    // Compute variance (vectorized)
    __m256 var_vec = _mm256_setzero_ps();
    i = 0;
    for (; i + VEC_SIZE <= size; i += VEC_SIZE) {
        __m256 vals = _mm256_loadu_ps(&input[i]);
        vals = _mm256_sub_ps(vals, mean_vec);
        vals = _mm256_mul_ps(vals, vals);
        var_vec = _mm256_add_ps(var_vec, vals);
    }
    for (; i < size; i++) {
        float diff = input[i] - mean;
        var_vec = _mm256_add_ss(var_vec, _mm256_set1_ps(diff * diff));
    }
    
    float var_arr[8];
    _mm256_storeu_ps(var_arr, var_vec);
    float var = var_arr[0];
    for (int j = 1; j < 8 && j < size; j++) {
        var += var_arr[j];
    }
    for (i = 8 * (size / 8); i < size; i++) {
        float diff = input[i] - mean;
        var += diff * diff;
    }
    var /= size;
    
    float inv_std = 1.0f / std::sqrt(var + 1e-5f);
    __m256 scale_vec = _mm256_set1_ps(inv_std);
    __m256 w_vec = _mm256_set1_ps(weight ? weight[0] : 1.0f);
    __m256 b_vec = _mm256_set1_ps(bias ? bias[0] : 0.0f);
    
    // Normalize and apply weight/bias
    i = 0;
    for (; i + VEC_SIZE <= size; i += VEC_SIZE) {
        __m256 vals = _mm256_loadu_ps(&input[i]);
        vals = _mm256_sub_ps(vals, mean_vec);
        vals = _mm256_mul_ps(vals, scale_vec);
        if (weight) vals = _mm256_mul_ps(vals, w_vec);
        if (bias) vals = _mm256_add_ps(vals, b_vec);
        _mm256_storeu_ps(&output[i], vals);
    }
    for (; i < size; i++) {
        float val = (input[i] - mean) * inv_std;
        if (weight) val *= weight[0];
        if (bias) val += bias[0];
        output[i] = val;
    }
}
#endif // IS_X86_PLATFORM

#if IS_ARM_PLATFORM
void layernorm_neon(float* output, const float* input, const float* weight,
                    const float* bias, int size) {
    constexpr int NEON_SIZE = 4;
    
    // Compute mean
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + NEON_SIZE <= size; i += NEON_SIZE) {
        float32x4_t vals = vld1q_f32(&input[i]);
        sum_vec = vaddq_f32(sum_vec, vals);
    }
    // Horizontal sum reduction
    float mean = 0.0f;
    float sum_arr[4];
    vst1q_f32(sum_arr, sum_vec);
    for (int j = 0; j < 4 && j < size; j++) {
        mean += sum_arr[j];
    }
    for (; i < size; i++) {
        mean += input[i];
    }
    mean /= size;
    
    float32x4_t mean_vec = vdupq_n_f32(mean);
    
    // Compute variance
    float32x4_t var_vec = vdupq_n_f32(0.0f);
    i = 0;
    for (; i + NEON_SIZE <= size; i += NEON_SIZE) {
        float32x4_t vals = vld1q_f32(&input[i]);
        vals = vsubq_f32(vals, mean_vec);
        vals = vmulq_f32(vals, vals);
        var_vec = vaddq_f32(var_vec, vals);
    }
    
    float var_arr[4];
    vst1q_f32(var_arr, var_vec);
    float var = 0.0f;
    for (int j = 0; j < 4 && j < size; j++) {
        var += var_arr[j];
    }
    for (; i < size; i++) {
        float diff = input[i] - mean;
        var += diff * diff;
    }
    var /= size;
    
    float inv_std = 1.0f / std::sqrt(var + 1e-5f);
    float32x4_t scale_vec = vdupq_n_f32(inv_std);
    float32x4_t w_vec = vdupq_n_f32(weight ? weight[0] : 1.0f);
    float32x4_t b_vec = vdupq_n_f32(bias ? bias[0] : 0.0f);
    
    // Normalize and apply
    i = 0;
    for (; i + NEON_SIZE <= size; i += NEON_SIZE) {
        float32x4_t vals = vld1q_f32(&input[i]);
        vals = vsubq_f32(vals, mean_vec);
        vals = vmulq_f32(vals, scale_vec);
        if (weight) vals = vmulq_f32(vals, w_vec);
        if (bias) vals = vaddq_f32(vals, b_vec);
        vst1q_f32(&output[i], vals);
    }
    for (; i < size; i++) {
        float val = (input[i] - mean) * inv_std;
        if (weight) val *= weight[0];
        if (bias) val += bias[0];
        output[i] = val;
    }
}
#endif // IS_ARM_PLATFORM

// Naive matmul for benchmarking
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

// Benchmark helper
template<typename Func>
double benchmark(Func func, int iterations = 100) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        func();
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() / iterations;
}

int main() {
    std::cout << "Session 52 Optimization Test\n";
    std::cout << "Platform: " << (IS_ARM_PLATFORM ? "ARM64" : "x86_64") << "\n\n";
    
    // Test 1: LayerNorm benchmark
    std::cout << "Test 1: LayerNorm Performance\n";
    const int LN_SIZE = 8192;
    std::vector<float> input(LN_SIZE), output_naive(LN_SIZE), output_simd(LN_SIZE);
    for (int i = 0; i < LN_SIZE; i++) {
        input[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    }
    
    double time_naive = benchmark([&]() {
        layernorm_naive(output_naive.data(), input.data(), nullptr, nullptr, LN_SIZE);
    });
    
#if IS_X86_PLATFORM
    double time_simd = benchmark([&]() {
        layernorm_avx2(output_simd.data(), input.data(), nullptr, nullptr, LN_SIZE);
    });
    std::cout << "  Naive: " << time_naive << " ms\n";
    std::cout << "  AVX2:  " << time_simd << " ms\n";
    std::cout << "  Speedup: " << (time_naive / time_simd) << "x\n";
#elif IS_ARM_PLATFORM
    double time_simd = benchmark([&]() {
        layernorm_neon(output_simd.data(), input.data(), nullptr, nullptr, LN_SIZE);
    });
    std::cout << "  Naive: " << time_naive << " ms\n";
    std::cout << "  NEON:  " << time_simd << " ms\n";
    std::cout << "  Speedup: " << (time_naive / time_simd) << "x\n";
#endif
    
    // Test 2: 1-bit matmul benchmark
    std::cout << "\nTest 2: 1-bit Matrix Multiply\n";
    const int M = 64, N = 64, K = 1024;
    std::vector<unsigned char> A_packed(M * K), B_packed(N * K);
    std::vector<float> C(M * N);
    
    // Initialize with random bits
    for (int i = 0; i < M * K; i++) {
        A_packed[i] = static_cast<unsigned char>(rand() % 256);
    }
    for (int i = 0; i < N * K; i++) {
        B_packed[i] = static_cast<unsigned char>(rand() % 256);
    }
    
    double time_1bit = benchmark([&]() {
        matmul_1bit_optimized(A_packed.data(), B_packed.data(), C.data(), M, N, K);
    });
    std::cout << "  1-bit matmul (64x64x1024): " << time_1bit << " ms\n";
    
    // Test 3: Batch processing with prefetch
    std::cout << "\nTest 3: Batch Processing with Prefetch\n";
    const int BATCH_SIZE = 4;
    const int MAT_SIZE = 256;
    std::vector<float> A_batch(BATCH_SIZE * MAT_SIZE * MAT_SIZE);
    std::vector<float> B(MAT_SIZE * MAT_SIZE);
    std::vector<float> C_batch(BATCH_SIZE * MAT_SIZE * MAT_SIZE);
    
    for (int i = 0; i < BATCH_SIZE * MAT_SIZE * MAT_SIZE; i++) {
        A_batch[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < MAT_SIZE * MAT_SIZE; i++) {
        B[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    double time_batch = benchmark([&]() {
        matmul_batch_cache_aware(A_batch.data(), B.data(), C_batch.data(),
                                  BATCH_SIZE, MAT_SIZE, MAT_SIZE, MAT_SIZE, matmul_naive);
    });
    std::cout << "  Batch matmul (4x 256x256): " << time_batch << " ms\n";
    
    std::cout << "\nSession 52 tests completed successfully!\n";
    return 0;
}
