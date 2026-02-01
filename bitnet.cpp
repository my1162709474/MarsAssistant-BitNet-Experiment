/**
 * BitNet: 1-bit Transformer Networks
 * Performance Optimized Implementation
 * Platform: Cross-platform (x86_64 AVX2/AVX-512 + ARM64 NEON)
 */

#include <cmath>
#include <cstring>
#include <cfloat>
#include <pthread.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <thread>

// Platform-specific SIMD headers - wrapped in proper guards
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#elif defined(__aarch64__) || defined(__arm__)
#include <arm_neon.h>
#endif

// Track platform capabilities for conditional compilation
#if defined(__x86_64__) || defined(__i386__)
#define IS_X86_PLATFORM 1
#else
#define IS_X86_PLATFORM 0
#endif

#if defined(__aarch64__) || defined(__arm__) || defined(__ARM_NEON)
#define IS_ARM_PLATFORM 1
#else
#define IS_ARM_PLATFORM 0
#endif

// Configuration
constexpr int BLOCK_SIZE = 64;
constexpr int CACHE_LINE_SIZE = 64;
constexpr int NEON_SIZE = 4;

// Compiler Optimization Hints
#ifdef __GNUC__
#define HOT_FUNC __attribute__((hot))
#define ALIGNED __attribute__((aligned(32)))
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#define FORCE_INLINE inline __attribute__((always_inline))
#define PREFETCH_READ(addr) __builtin_prefetch((addr), 0, 3)
#define PREFETCH_WRITE(addr) __builtin_prefetch((addr), 1, 3)
#define RESTRICT __restrict__
#else
#define HOT_FUNC
#define ALIGNED
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#define FORCE_INLINE inline
#define PREFETCH_READ(addr)
#define PREFETCH_WRITE(addr)
#define RESTRICT
#endif

// Forward declarations
void matmul_multi_level_blocked(const float* A, const float* B, float* C, int M, int N, int K);
void matmul_batch(const float* A_batch, const float* B, float* C_batch,
                  int batch_size, int M, int N, int K);

// ==================== Data Structures ====================

struct Matrix {
    float* data;
    int rows;
    int cols;
    int stride;
    
    Matrix(int r = 0, int c = 0) : rows(r), cols(c), stride(c) {
        posix_memalign(reinterpret_cast<void**>(&data), CACHE_LINE_SIZE, 
                       sizeof(float) * rows * cols);
        std::memset(data, 0, sizeof(float) * rows * cols);
    }
    
    ~Matrix() {
        free(data);
    }
};

struct BitMatrix {
    unsigned char* data;
    int rows;
    int cols;
    int stride_bytes;
    
    BitMatrix(int r = 0, int c = 0) : rows(r), cols(c) {
        stride_bytes = (cols + 7) / 8;
        posix_memalign(reinterpret_cast<void**>(&data), CACHE_LINE_SIZE,
                       sizeof(unsigned char) * rows * stride_bytes);
        std::memset(data, 0, sizeof(unsigned char) * rows * stride_bytes);
    }
    
    ~BitMatrix() {
        free(data);
    }
    
    void pack_from_float(const float* src) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (src[i * cols + j] > 0.0f) {
                    data[i * stride_bytes + j / 8] |= (1 << (j % 8));
                }
            }
        }
    }
    
#if IS_ARM_PLATFORM
    void pack_from_float_neon(const float* src);
#endif
};

// ==================== Session 50: ARM NEON Ultra Optimizations ====================
// Date: 2026-02-01 17:51

// 1. Ultra-Fast NEON Matrix Multiply (Apple Silicon Optimized)
#if IS_ARM_PLATFORM

void matmul_neon(const float* A, const float* B, float* C,
                 int M, int N, int K) {
    constexpr int NEON_SIZE = 4;
    
    for (int i = 0; i < M; i++) {
        const float* A_row = A + i * K;
        float* C_row = C + i * N;
        
        float32x4_t c_vec[64];
        int num_vec = N / NEON_SIZE;
        
        // Initialize accumulators
        for (int j = 0; j < num_vec; j++) {
            c_vec[j] = vdupq_n_f32(0.0f);
        }
        
        for (int k = 0; k < K; k++) {
            float32x4_t a_val = vdupq_n_f32(A_row[k]);
            const float* B_k = B + k * N;
            
            // Prefetch next iteration
            if (k + 4 < K) {
                PREFETCH_READ(&A_row[k + 4]);
                PREFETCH_READ(&B_k[0]);
            }
            
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

// Alias for compatibility
void matmul_avx2(const float* A, const float* B, float* C,
                 int M, int N, int K) {
    matmul_neon(A, B, C, M, N, K);
}

#endif // IS_ARM_PLATFORM

// ==================== Session 51: x86 AVX2 Ultra Optimizations ====================
// Date: 2026-02-01 18:10

#if IS_X86_PLATFORM

// 1. AVX2 Matrix Multiply with 8x Unrolling
void matmul_avx2_8x_unroll(const float* A, const float* B, float* C,
                           int M, int N, int K) {
    constexpr int AVX_SIZE = 8;
    constexpr int UNROLL_FACTOR = 8;

    for (int i = 0; i < M; i++) {
        const float* A_row = A + i * K;
        float* C_row = C + i * N;

        int num_vec = N / AVX_SIZE;
        int unrolled = (num_vec / UNROLL_FACTOR) * UNROLL_FACTOR;

        // Initialize output
        for (int j = 0; j < unrolled * AVX_SIZE; j += UNROLL_FACTOR * AVX_SIZE) {
            for (int u = 0; u < UNROLL_FACTOR; u++) {
                _mm256_storeu_ps(&C_row[j + u * AVX_SIZE], _mm256_setzero_ps());
            }
        }
        for (int j = unrolled * AVX_SIZE; j < N; j++) {
            C_row[j] = 0.0f;
        }

        // Main loop with 8x unrolling
        for (int k = 0; k < K; k++) {
            __m256 a_val = _mm256_set1_ps(A_row[k]);
            const float* B_k = B + k * N;

            // Prefetch
            if (k + 4 < K) {
                PREFETCH_READ(&A_row[k + 4]);
            }

            for (int j = 0; j < unrolled; j += UNROLL_FACTOR) {
                // Load 8 AVX vectors
                __m256 b0 = _mm256_loadu_ps(&B_k[(j + 0) * AVX_SIZE]);
                __m256 b1 = _mm256_loadu_ps(&B_k[(j + 1) * AVX_SIZE]);
                __m256 b2 = _mm256_loadu_ps(&B_k[(j + 2) * AVX_SIZE]);
                __m256 b3 = _mm256_loadu_ps(&B_k[(j + 3) * AVX_SIZE]);
                __m256 b4 = _mm256_loadu_ps(&B_k[(j + 4) * AVX_SIZE]);
                __m256 b5 = _mm256_loadu_ps(&B_k[(j + 5) * AVX_SIZE]);
                __m256 b6 = _mm256_loadu_ps(&B_k[(j + 6) * AVX_SIZE]);
                __m256 b7 = _mm256_loadu_ps(&B_k[(j + 7) * AVX_SIZE]);

                // Load accumulators
                __m256 c0 = _mm256_loadu_ps(&C_row[(j + 0) * AVX_SIZE]);
                __m256 c1 = _mm256_loadu_ps(&C_row[(j + 1) * AVX_SIZE]);
                __m256 c2 = _mm256_loadu_ps(&C_row[(j + 2) * AVX_SIZE]);
                __m256 c3 = _mm256_loadu_ps(&C_row[(j + 3) * AVX_SIZE]);
                __m256 c4 = _mm256_loadu_ps(&C_row[(j + 4) * AVX_SIZE]);
                __m256 c5 = _mm256_loadu_ps(&C_row[(j + 5) * AVX_SIZE]);
                __m256 c6 = _mm256_loadu_ps(&C_row[(j + 6) * AVX_SIZE]);
                __m256 c7 = _mm256_loadu_ps(&C_row[(j + 7) * AVX_SIZE]);

                // FMA operations (fused multiply-add)
                c0 = _mm256_fmadd_ps(a_val, b0, c0);
                c1 = _mm256_fmadd_ps(a_val, b1, c1);
                c2 = _mm256_fmadd_ps(a_val, b2, c2);
                c3 = _mm256_fmadd_ps(a_val, b3, c3);
                c4 = _mm256_fmadd_ps(a_val, b4, c4);
                c5 = _mm256_fmadd_ps(a_val, b5, c5);
                c6 = _mm256_fmadd_ps(a_val, b6, c6);
                c7 = _mm256_fmadd_ps(a_val, b7, c7);

                // Store
                _mm256_storeu_ps(&C_row[(j + 0) * AVX_SIZE], c0);
                _mm256_storeu_ps(&C_row[(j + 1) * AVX_SIZE], c1);
                _mm256_storeu_ps(&C_row[(j + 2) * AVX_SIZE], c2);
                _mm256_storeu_ps(&C_row[(j + 3) * AVX_SIZE], c3);
                _mm256_storeu_ps(&C_row[(j + 4) * AVX_SIZE], c4);
                _mm256_storeu_ps(&C_row[(j + 5) * AVX_SIZE], c5);
                _mm256_storeu_ps(&C_row[(j + 6) * AVX_SIZE], c6);
                _mm256_storeu_ps(&C_row[(j + 7) * AVX_SIZE], c7);
            }
        }
    }
}

// 2. AVX2 Vectorized ReLU
void relu_avx2(float* data, int size) {
    constexpr int AVX_SIZE = 8;
    const __m256 zero = _mm256_setzero_ps();

    for (int i = 0; i + AVX_SIZE <= size; i += AVX_SIZE) {
        __m256 vals = _mm256_loadu_ps(&data[i]);
        vals = _mm256_max_ps(vals, zero);
        _mm256_storeu_ps(&data[i], vals);
    }

    // Handle remainder
    for (int i = size - (size % AVX_SIZE); i < size; i++) {
        data[i] = std::max(0.0f, data[i]);
    }
}

// 3. AVX2 Vectorized GELU (7-term polynomial)
FORCE_INLINE float gelu_avx2_poly(float x) {
    const float a0 = 0.99999988f;
    const float a1 = 0.99996151f;
    const float a2 = 0.24991058f;
    const float a3 = 0.03324052f;
    const float a4 = 0.00358906f;
    const float a5 = 0.00025026f;
    const float a6 = 0.00000693f;

    float x2 = x * x;
    float x3 = x2 * x;
    float x4 = x2 * x2;
    float x5 = x4 * x;
    float x6 = x4 * x2;
    float x7 = x6 * x;

    return a0 * x - a1 * x3 / 6.0f + a2 * x5 / 120.0f - a3 * x7 / 5040.0f
           + a4 * x2 / 24.0f - a5 * x4 / 720.0f + a6 * x6 / 40320.0f;
}

void gelu_avx2_vectorized(float* data, int size) {
    constexpr int AVX_SIZE = 8;

    for (int i = 0; i + AVX_SIZE <= size; i += AVX_SIZE) {
        __m256 x = _mm256_loadu_ps(&data[i]);
        float x_vals[8];
        _mm256_storeu_ps(x_vals, x);

        for (int j = 0; j < 8 && i + j < size; j++) {
            x_vals[j] = gelu_avx2_poly(x_vals[j]);
        }

        _mm256_storeu_ps(&data[i], _mm256_loadu_ps(x_vals));
    }

    for (int i = size - (size % AVX_SIZE); i < size; i++) {
        data[i] = gelu_avx2_poly(data[i]);
    }
}

// 4. AVX2 Hyper-Parallel Softmax
void softmax_avx2_hyper(float* data, int size) {
    constexpr int AVX_SIZE = 8;

    // Find max (vectorized)
    __m256 max_vec = _mm256_set1_ps(-FLT_MAX);
    for (int i = 0; i + AVX_SIZE <= size; i += AVX_SIZE) {
        __m256 vals = _mm256_loadu_ps(&data[i]);
        max_vec = _mm256_max_ps(max_vec, vals);
    }

    // Horizontal max reduction
    float row_max = max_vec[0];
    for (int i = 8; i < size; i++) {
        row_max = std::max(row_max, data[i]);
    }

    max_vec = _mm256_set1_ps(row_max);

    // Exp and sum (vectorized)
    __m256 sum_vec = _mm256_setzero_ps();
    for (int i = 0; i + AVX_SIZE <= size; i += AVX_SIZE) {
        __m256 vals = _mm256_loadu_ps(&data[i]);
        vals = _mm256_sub_ps(vals, max_vec);
        vals = _mm256_exp_ps(vals);
        sum_vec = _mm256_add_ps(sum_vec, vals);
        _mm256_storeu_ps(&data[i], vals);
    }

    // Horizontal sum
    float row_sum = 0.0f;
    float sum_arr[8];
    _mm256_storeu_ps(sum_arr, sum_vec);
    for (int j = 0; j < 8; j++) {
        row_sum += sum_arr[j];
    }
    for (int i = 8; i < size; i++) {
        row_sum += data[i];
    }

    // Normalize
    float inv_sum = 1.0f / (row_sum + 1e-8f);
    __m256 inv_vec = _mm256_set1_ps(inv_sum);

    for (int i = 0; i + AVX_SIZE <= size; i += AVX_SIZE) {
        __m256 vals = _mm256_loadu_ps(&data[i]);
        vals = _mm256_mul_ps(vals, inv_vec);
        _mm256_storeu_ps(&data[i], vals);
    }

    for (int i = size - (size % AVX_SIZE); i < size; i++) {
        data[i] *= inv_sum;
    }
}

// 5. AVX2 Fused LayerNorm + GELU
void fused_layernorm_gelu_avx2(float* data, const float* weight,
                                const float* bias, int size) {
    // Compute mean
    __m256 sum_vec = _mm256_setzero_ps();
    for (int i = 0; i + AVX_SIZE <= size; i += AVX_SIZE) {
        __m256 vals = _mm256_loadu_ps(&data[i]);
        sum_vec = _mm256_add_ps(sum_vec, vals);
    }

    float mean = 0.0f;
    float sum_arr[8];
    _mm256_storeu_ps(sum_arr, sum_vec);
    for (int j = 0; j < 8; j++) {
        mean += sum_arr[j];
    }
    for (int i = 8; i < size; i++) {
        mean += data[i];
    }
    mean /= size;

    // Compute variance
    __m256 mean_vec = _mm256_set1_ps(mean);
    __m256 var_vec = _mm256_setzero_ps();
    for (int i = 0; i + AVX_SIZE <= size; i += AVX_SIZE) {
        __m256 vals = _mm256_loadu_ps(&data[i]);
        vals = _mm256_sub_ps(vals, mean_vec);
        vals = _mm256_mul_ps(vals, vals);
        var_vec = _mm256_add_ps(var_vec, vals);
    }

    float var = 0.0f;
    _mm256_storeu_ps(sum_arr, var_vec);
    for (int j = 0; j < 8; j++) {
        var += sum_arr[j];
    }
    for (int i = 8; i < size; i++) {
        float diff = data[i] - mean;
        var += diff * diff;
    }
    var /= size;

    // Normalize, apply GELU, weight, and bias
    float inv_std = 1.0f / std::sqrt(var + 1e-5f);
    __m256 scale_vec = _mm256_set1_ps(inv_std);
    __m256 w_vec = _mm256_set1_ps(weight ? weight[0] : 1.0f);
    __m256 b_vec = _mm256_set1_ps(bias ? bias[0] : 0.0f);

    for (int i = 0; i + AVX_SIZE <= size; i += AVX_SIZE) {
        __m256 vals = _mm256_loadu_ps(&data[i]);
        vals = _mm256_sub_ps(vals, mean_vec);
        vals = _mm256_mul_ps(vals, scale_vec);

        float vals_f[8];
        _mm256_storeu_ps(vals_f, vals);
        for (int j = 0; j < 8 && i + j < size; j++) {
            vals_f[j] = gelu_avx2_poly(vals_f[j]);
        }
        vals = _mm256_loadu_ps(vals_f);

        if (weight) {
            vals = _mm256_mul_ps(vals, w_vec);
        }
        if (bias) {
            vals = _mm256_add_ps(vals, b_vec);
        }

        _mm256_storeu_ps(&data[i], vals);
    }
}

#endif // IS_X86_PLATFORM

// 2. NEON Vectorized ReLU Activation
#if IS_ARM_PLATFORM

void relu_neon(float* data, int size) {
    constexpr int NEON_SIZE = 4;
    const float32x4_t zero = vdupq_n_f32(0.0f);
    
    for (int i = 0; i + NEON_SIZE <= size; i += NEON_SIZE) {
        float32x4_t vals = vld1q_f32(&data[i]);
        vals = vmaxq_f32(vals, zero);
        vst1q_f32(&data[i], vals);
    }
    
    // Handle remainder
    for (int i = size - (size % NEON_SIZE); i < size; i++) {
        data[i] = std::max(0.0f, data[i]);
    }
}

#endif // IS_ARM_PLATFORM

// 3. NEON 8x Loop Unrolling (Apple Silicon M-series)
#if IS_ARM_PLATFORM

void matmul_neon_8x_unroll(const float* A, const float* B, float* C,
                           int M, int N, int K) {
    constexpr int NEON_SIZE = 4;
    constexpr int UNROLL_FACTOR = 8;  // 32 floats per iteration
    
    for (int i = 0; i < M; i++) {
        const float* A_row = A + i * K;
        float* C_row = C + i * N;
        
        int num_vec = N / NEON_SIZE;
        int unrolled = (num_vec / UNROLL_FACTOR) * UNROLL_FACTOR;
        
        // Initialize output
        for (int j = 0; j < unrolled * NEON_SIZE; j += UNROLL_FACTOR * NEON_SIZE) {
            for (int u = 0; u < UNROLL_FACTOR; u++) {
                vst1q_f32(&C_row[j + u * NEON_SIZE], vdupq_n_f32(0.0f));
            }
        }
        for (int j = unrolled * NEON_SIZE; j < N; j++) {
            C_row[j] = 0.0f;
        }
        
        // Main loop with 8x unrolling
        for (int k = 0; k < K; k++) {
            float32x4_t a_val = vdupq_n_f32(A_row[k]);
            const float* B_k = B + k * N;
            
            // Prefetch
            if (k + 4 < K) {
                PREFETCH_READ(&A_row[k + 4]);
            }
            
            for (int j = 0; j < unrolled; j += UNROLL_FACTOR) {
                // Load 8 NEON vectors
                float32x4_t b0 = vld1q_f32(&B_k[(j + 0) * NEON_SIZE]);
                float32x4_t b1 = vld1q_f32(&B_k[(j + 1) * NEON_SIZE]);
                float32x4_t b2 = vld1q_f32(&B_k[(j + 2) * NEON_SIZE]);
                float32x4_t b3 = vld1q_f32(&B_k[(j + 3) * NEON_SIZE]);
                float32x4_t b4 = vld1q_f32(&B_k[(j + 4) * NEON_SIZE]);
                float32x4_t b5 = vld1q_f32(&B_k[(j + 5) * NEON_SIZE]);
                float32x4_t b6 = vld1q_f32(&B_k[(j + 6) * NEON_SIZE]);
                float32x4_t b7 = vld1q_f32(&B_k[(j + 7) * NEON_SIZE]);
                
                // Load accumulators
                float32x4_t c0 = vld1q_f32(&C_row[(j + 0) * NEON_SIZE]);
                float32x4_t c1 = vld1q_f32(&C_row[(j + 1) * NEON_SIZE]);
                float32x4_t c2 = vld1q_f32(&C_row[(j + 2) * NEON_SIZE]);
                float32x4_t c3 = vld1q_f32(&C_row[(j + 3) * NEON_SIZE]);
                float32x4_t c4 = vld1q_f32(&C_row[(j + 4) * NEON_SIZE]);
                float32x4_t c5 = vld1q_f32(&C_row[(j + 5) * NEON_SIZE]);
                float32x4_t c6 = vld1q_f32(&C_row[(j + 6) * NEON_SIZE]);
                float32x4_t c7 = vld1q_f32(&C_row[(j + 7) * NEON_SIZE]);
                
                // FMA operations
                c0 = vfmaq_f32(c0, a_val, b0);
                c1 = vfmaq_f32(c1, a_val, b1);
                c2 = vfmaq_f32(c2, a_val, b2);
                c3 = vfmaq_f32(c3, a_val, b3);
                c4 = vfmaq_f32(c4, a_val, b4);
                c5 = vfmaq_f32(c5, a_val, b5);
                c6 = vfmaq_f32(c6, a_val, b6);
                c7 = vfmaq_f32(c7, a_val, b7);
                
                // Store
                vst1q_f32(&C_row[(j + 0) * NEON_SIZE], c0);
                vst1q_f32(&C_row[(j + 1) * NEON_SIZE], c1);
                vst1q_f32(&C_row[(j + 2) * NEON_SIZE], c2);
                vst1q_f32(&C_row[(j + 3) * NEON_SIZE], c3);
                vst1q_f32(&C_row[(j + 4) * NEON_SIZE], c4);
                vst1q_f32(&C_row[(j + 5) * NEON_SIZE], c5);
                vst1q_f32(&C_row[(j + 6) * NEON_SIZE], c6);
                vst1q_f32(&C_row[(j + 7) * NEON_SIZE], c7);
            }
        }
    }
}

#endif // IS_ARM_PLATFORM

// 4. NEON Vectorized GELU Approximation (7-term polynomial)
#if IS_ARM_PLATFORM

float gelu_neon_poly(float x) {
    // 7-term polynomial approximation for GELU
    // Based on Taylor series expansion
    const float a0 = 0.99999988f;
    const float a1 = 0.99996151f;
    const float a2 = 0.24991058f;
    const float a3 = 0.03324052f;
    const float a4 = 0.00358906f;
    const float a5 = 0.00025026f;
    const float a6 = 0.00000693f;
    
    float x2 = x * x;
    float x3 = x2 * x;
    float x4 = x2 * x2;
    float x5 = x4 * x;
    float x6 = x4 * x2;
    float x7 = x6 * x;
    
    return a0 * x - a1 * x3 / 6.0f + a2 * x5 / 120.0f - a3 * x7 / 5040.0f
           + a4 * x2 / 24.0f - a5 * x4 / 720.0f + a6 * x6 / 40320.0f;
}

void gelu_neon_vectorized(float* data, int size) {
    constexpr int NEON_SIZE = 4;
    
    for (int i = 0; i + NEON_SIZE <= size; i += NEON_SIZE) {
        float32x4_t x = vld1q_f32(&data[i]);
        
        // Extract elements for polynomial approximation
        float x_vals[4];
        vst1q_f32(x_vals, x);
        
        float32x4_t result = vdupq_n_f32(0.0f);
        for (int j = 0; j < 4; j++) {
            float32x4_t elem = vdupq_n_f32(gelu_neon_poly(x_vals[j]));
            result = vaddq_f32(result, vdupq_n_f32(0.0f));  // Placeholder
        }
        
        // Use scalar fallback for accuracy (polynomial is element-wise)
        for (int j = 0; j < 4 && i + j < size; j++) {
            data[i + j] = gelu_neon_poly(data[i + j]);
        }
    }
    
    // Remainder
    for (int i = size - (size % NEON_SIZE); i < size; i++) {
        data[i] = gelu_neon_poly(data[i]);
    }
}

#endif // IS_ARM_PLATFORM

// 5. NEON Vectorized Sigmoid with LUT
#if IS_ARM_PLATFORM

static float sigmoid_lut[256];

void init_sigmoid_lut_neon() {
    for (int i = 0; i < 256; i++) {
        float x = (i - 128) / 16.0f;  // Range [-8, 8]
        sigmoid_lut[i] = 1.0f / (1.0f + std::exp(-x));
    }
}

void sigmoid_neon_lut(float* data, int size) {
    constexpr int NEON_SIZE = 4;
    
    for (int i = 0; i + NEON_SIZE <= size; i += NEON_SIZE) {
        float32x4_t x = vld1q_f32(&data[i]);
        
        // Clamp to LUT range
        float32x4_t clamped = vmaxq_f32(vminq_f32(x, vdupq_n_f32(7.99f)), vdupq_n_f32(-7.99f));
        
        // Linear interpolation LUT lookup
        float x_vals[4];
        vst1q_f32(x_vals, clamped);
        
        for (int j = 0; j < 4 && i + j < size; j++) {
            float val = x_vals[j];
            int idx = static_cast<int>((val + 8.0f) * 16.0f);
            float frac = (val + 8.0f) * 16.0f - idx;
            data[i + j] = sigmoid_lut[idx] * (1.0f - frac) + sigmoid_lut[idx + 1] * frac;
        }
    }
    
    for (int i = size - (size % NEON_SIZE); i < size; i++) {
        float x = std::max(-8.0f, std::min(8.0f, data[i]));
        int idx = static_cast<int>((x + 8.0f) * 16.0f);
        float frac = (x + 8.0f) * 16.0f - idx;
        data[i] = sigmoid_lut[idx] * (1.0f - frac) + sigmoid_lut[idx + 1] * frac;
    }
}

#endif // IS_ARM_PLATFORM

// 6. NEON Hyper-Parallel Softmax
#if IS_ARM_PLATFORM

void softmax_neon_hyper(float* data, int size) {
    constexpr int NEON_SIZE = 4;
    
    // Find max (vectorized)
    float32x4_t max_vec = vdupq_n_f32(-FLT_MAX);
    for (int i = 0; i + NEON_SIZE <= size; i += NEON_SIZE) {
        float32x4_t vals = vld1q_f32(&data[i]);
        max_vec = vmaxq_f32(max_vec, vals);
    }
    
    // Horizontal max
    float row_max = vgetq_lane_f32(max_vec, 0);
    for (int i = 4; i < size; i++) {
        row_max = std::max(row_max, data[i]);
    }
    
    max_vec = vdupq_n_f32(row_max);
    
    // Exp and sum (vectorized)
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    for (int i = 0; i + NEON_SIZE <= size; i += NEON_SIZE) {
        float32x4_t vals = vld1q_f32(&data[i]);
        vals = vsubq_f32(vals, max_vec);
        
        // exp approximation
        float vals_f[4];
        vst1q_f32(vals_f, vals);
        for (int j = 0; j < 4 && i + j < size; j++) {
            vals_f[j] = std::exp(vals_f[j]);
        }
        vals = vld1q_f32(vals_f);
        
        sum_vec = vaddq_f32(sum_vec, vals);
        vst1q_f32(&data[i], vals);
    }
    
    // Horizontal sum
    float row_sum = vgetq_lane_f32(sum_vec, 0);
    for (int i = 4; i < size; i++) {
        row_sum += data[i];
    }
    
    // Normalize
    float inv_sum = 1.0f / (row_sum + 1e-8f);
    float32x4_t inv_vec = vdupq_n_f32(inv_sum);
    
    for (int i = 0; i + NEON_SIZE <= size; i += NEON_SIZE) {
        float32x4_t vals = vld1q_f32(&data[i]);
        vals = vmulq_f32(vals, inv_vec);
        vst1q_f32(&data[i], vals);
    }
    
    for (int i = size - (size % NEON_SIZE); i < size; i++) {
        data[i] *= inv_sum;
    }
}

#endif // IS_ARM_PLATFORM

// 7. NEON Fused LayerNorm + GELU
#if IS_ARM_PLATFORM

void fused_layernorm_gelu_neon(float* data, const float* weight, 
                                const float* bias, int size) {
    // Compute mean
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    for (int i = 0; i + NEON_SIZE <= size; i += NEON_SIZE) {
        float32x4_t vals = vld1q_f32(&data[i]);
        sum_vec = vaddq_f32(sum_vec, vals);
    }
    
    float mean = vgetq_lane_f32(sum_vec, 0);
    for (int i = 4; i < size; i++) {
        mean += data[i];
    }
    mean /= size;
    
    // Compute variance
    float32x4_t mean_vec = vdupq_n_f32(mean);
    float32x4_t var_vec = vdupq_n_f32(0.0f);
    for (int i = 0; i + NEON_SIZE <= size; i += NEON_SIZE) {
        float32x4_t vals = vld1q_f32(&data[i]);
        vals = vsubq_f32(vals, mean_vec);
        vals = vmulq_f32(vals, vals);
        var_vec = vaddq_f32(var_vec, vals);
    }
    
    float var = vgetq_lane_f32(var_vec, 0);
    for (int i = 4; i < size; i++) {
        float diff = data[i] - mean;
        var += diff * diff;
    }
    var /= size;
    
    // Normalize, apply GELU, weight, and bias
    float inv_std = 1.0f / std::sqrt(var + 1e-5f);
    float32x4_t scale_vec = vdupq_n_f32(inv_std);
    float32x4_t w_vec = vdupq_n_f32(weight ? weight[0] : 1.0f);
    float32x4_t b_vec = vdupq_n_f32(bias ? bias[0] : 0.0f);
    
    for (int i = 0; i + NEON_SIZE <= size; i += NEON_SIZE) {
        // LayerNorm
        float32x4_t vals = vld1q_f32(&data[i]);
        vals = vsubq_f32(vals, mean_vec);
        vals = vmulq_f32(vals, scale_vec);
        
        // GELU approximation
        float vals_f[4];
        vst1q_f32(vals_f, vals);
        for (int j = 0; j < 4 && i + j < size; j++) {
            vals_f[j] = gelu_neon_poly(vals_f[j]);
        }
        vals = vld1q_f32(vals_f);
        
        // Weight and bias
        if (weight) {
            vals = vmulq_f32(vals, w_vec);
        }
        if (bias) {
            vals = vaddq_f32(vals, b_vec);
        }
        
        vst1q_f32(&data[i], vals);
    }
}

#endif // IS_ARM_PLATFORM

// ==================== Cross-Platform Fallbacks ====================

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

void matmul_blocked(const float* A, const float* B, float* C,
                    int M, int N, int K) {
    for (int i = 0; i < M; i += BLOCK_SIZE) {
        for (int j = 0; j < N; j += BLOCK_SIZE) {
            for (int k = 0; k < K; k += BLOCK_SIZE) {
                for (int ii = i; ii < std::min(i + BLOCK_SIZE, M); ii++) {
                    const float* A_block = &A[ii * K + k];
                    float* C_block = &C[ii * N + j];
                    
                    if (ii + 4 < std::min(i + BLOCK_SIZE, M)) {
                        PREFETCH_READ(&A[(ii + 4) * K + k]);
                    }
                    
                    for (int jj = j; jj < std::min(j + BLOCK_SIZE, N); jj++) {
                        float sum = 0.0f;
                        for (int kk = k; kk < std::min(k + BLOCK_SIZE, K); kk++) {
                            sum += A_block[kk - k] * B[kk * N + jj];
                        }
                        C_block[jj - j] += sum;
                    }
                }
            }
        }
    }
}

// ==================== 1-bit Quantization ====================

void quantize_1bit(const float* input, unsigned char* output, int size, float threshold) {
#if IS_ARM_PLATFORM
    constexpr int NEON_SIZE = 4;
    const float32x4_t thresh_vec = vdupq_n_f32(threshold);
    
    for (int i = 0; i + NEON_SIZE <= size; i += NEON_SIZE) {
        float32x4_t vals = vld1q_f32(&input[i]);
        uint32x4_t cmp = vcgtq_f32(vals, thresh_vec);
        unsigned mask = vgetq_lane_u32(cmp, 0) | (vgetq_lane_u32(cmp, 1) << 1) |
                        (vgetq_lane_u32(cmp, 2) << 2) | (vgetq_lane_u32(cmp, 3) << 3);
        output[i] = mask & 0xFF;
    }
    for (int i = size - (size % NEON_SIZE); i < size; i++) {
        output[i] = (input[i] > threshold) ? 1 : 0;
    }
#elif IS_X86_PLATFORM
    constexpr int AVX_SIZE = 8;
    const __m256 thresh_vec = _mm256_set1_ps(threshold);
    
    for (int i = 0; i + AVX_SIZE <= size; i += AVX_SIZE) {
        __m256 vals = _mm256_loadu_ps(&input[i]);
        __m256 cmp = _mm256_cmp_ps(vals, thresh_vec, _CMP_GT_OQ);
        unsigned mask = _mm256_movemask_ps(cmp);
        output[i] = (mask & 1) | ((mask & 2) << 1) | ((mask & 4) << 2) | ((mask & 8) << 3) |
                    ((mask & 16) << 4) | ((mask & 32) << 5) | ((mask & 64) << 6) | ((mask & 128) << 7);
    }
    for (int i = size - (size % AVX_SIZE); i < size; i++) {
        output[i] = (input[i] > threshold) ? 1 : 0;
    }
#else
    for (int i = 0; i < size; i++) {
        output[i] = (input[i] > threshold) ? 1 : 0;
    }
#endif
}

void matmul_1bit_packed(const unsigned char* A_packed, const unsigned char* B_packed, 
                        float* C, int M, int N, int K) {
    const int K_words = (K + 31) / 32;
    
    for (int i = 0; i < M; i++) {
        const unsigned int* A_words = reinterpret_cast<const unsigned int*>(A_packed + i * K);
        
        for (int j = 0; j < N; j++) {
            const unsigned int* B_words = reinterpret_cast<const unsigned int*>(B_packed + j * K);
            int popcount = 0;
            
            for (int w = 0; w < K_words; w++) {
                popcount += __builtin_popcount(A_words[w] ^ B_words[w]);
            }
            
            C[i * N + j] = static_cast<float>(K - 2 * popcount);
        }
    }
}

// ==================== Multi-threaded Parallel MatMul ====================

struct MatmulThreadData {
    const float* A;
    const float* B;
    float* C;
    int M, N, K;
    int start_row, end_row;
};

void* matmul_thread(void* arg) {
    MatmulThreadData* data = (MatmulThreadData*)arg;
    
#if IS_ARM_PLATFORM
    matmul_neon_8x_unroll(data->A + data->start_row * data->K,
                         data->B,
                         data->C + data->start_row * data->N,
                         data->end_row - data->start_row,
                         data->N,
                         data->K);
#elif IS_X86_PLATFORM
    matmul_avx2(data->A + data->start_row * data->K,
                data->B,
                data->C + data->start_row * data->N,
                data->end_row - data->start_row,
                data->N,
                data->K);
#else
    matmul_naive(data->A + data->start_row * data->K,
                 data->B,
                 data->C + data->start_row * data->N,
                 data->end_row - data->start_row,
                 data->N,
                 data->K);
#endif
    return nullptr;
}

void matmul_parallel(const float* A, const float* B, float* C, int M, int N, int K) {
    int num_threads = std::thread::hardware_concurrency();
    if (num_threads > M) num_threads = M;
    
    std::vector<pthread_t> threads(num_threads);
    std::vector<MatmulThreadData> thread_data(num_threads);
    
    int rows_per_thread = M / num_threads;
    int remaining = M % num_threads;
    
    for (int t = 0; t < num_threads; t++) {
        thread_data[t] = {A, B, C, M, N, K,
                         t * rows_per_thread,
                         (t + 1) * rows_per_thread + (t < remaining ? 1 : 0)};
        pthread_create(&threads[t], nullptr, matmul_thread, &thread_data[t]);
    }
    
    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], nullptr);
    }
}

// ==================== Batch Matrix Multiply ====================

void matmul_batch(const float* A_batch, const float* B, float* C_batch,
                  int batch_size, int M, int N, int K) {
    for (int b = 0; b < batch_size; b++) {
#if IS_ARM_PLATFORM
        matmul_neon_8x_unroll(A_batch + b * M * K, B, C_batch + b * M * N, M, N, K);
#elif IS_X86_PLATFORM
        matmul_avx2(A_batch + b * M * K, B, C_batch + b * M * N, M, N, K);
#else
        matmul_naive(A_batch + b * M * K, B, C_batch + b * M * N, M, N, K);
#endif
    }
}

// ==================== Sparse Matrix Operations ====================

void matmul_sparse_csr(const float* A_values, const int* A_col_idx, const int* A_row_ptr,
                       const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        int row_start = A_row_ptr[i];
        int row_end = A_row_ptr[i + 1];
        
        for (int nz = row_start; nz < row_end; nz++) {
            int k = A_col_idx[nz];
            float a_val = A_values[nz];
            const float* B_row = B + k * N;
            float* C_row = C + i * N;
            
            for (int j = 0; j < N; j++) {
                C_row[j] += a_val * B_row[j];
            }
        }
    }
}

// ==================== Main Benchmark ====================

int main() {
    std::cout << "BitNet Performance Optimization - Session 50\n";
    std::cout << "Platform: " << (IS_ARM_PLATFORM ? "ARM64 (Apple Silicon)" : "x86_64") << "\n";
    
#if IS_ARM_PLATFORM
    init_sigmoid_lut_neon();
    std::cout << "Optimizations:\n";
    std::cout << "  - NEON 8x unrolling\n";
    std::cout << "  - NEON GELU polynomial\n";
    std::cout << "  - NEON sigmoid LUT\n";
    std::cout << "  - NEON hyper softmax\n";
    std::cout << "  - Fused LayerNorm + GELU\n";
#endif
    
    std::cout << "\nCompilation successful!\n";
    return 0;
}

// ==================== ARM NEON Optimization ====================
#ifdef __ARM_NEON

void matmul_neon(const float* A, const float* B, float* C,
                 int M, int N, int K) {
    constexpr int NEON_SIZE = 4;  // 128-bit / 32-bit
    
    for (int i = 0; i < M; i++) {
        const float* A_row = A + i * K;
        float* C_row = C + i * N;
        
        int num_vec = N / NEON_SIZE;
        float32x4_t c_vec[128] = {};
        
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

// NEON dot product for 1-bit quantization
int dot_product_neon(const unsigned char* a, const unsigned char* b, int len) {
    int count = 0;
    int i = 0;
    
    for (; i + 15 < len; i += 16) {
        uint8x16_t va = vld1q_u8(a + i);
        uint8x16_t vb = vld1q_u8(b + i);
        
        // Population count
        uint8x16_t xored = veorq_u8(va, vb);
        uint8x16_t masked = vmvnq_u8(xored);
        
        // Sum bits (popcount)
        uint16x8_t sum1 = vpaddlq_u8(vpaddlq_u4(vpaddlq_u1(masked)));
        count += vgetq_lane_s16(sum1, 0) + vgetq_lane_s16(sum1, 4);
    }
    
    // Handle remainder
    for (; i < len; i++) {
        if ((a[i >> 3] >> (i & 7)) == (b[i >> 3] >> (i & 7))) {
            count++;
        }
    }
    
    return count;
}

#endif

// ==================== Session 52: Memory Access & Quantization Optimizations ====================
// Date: 2026-02-01 18:22

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
void matmul_batch_cache_aware(const float* A_batch, const float* B, float* C_batch,
                               int batch_size, int M, int N, int K) {
    constexpr int PREFETCH_STRIDE = 64;
    
    for (int b = 0; b < batch_size; b++) {
        const float* A = A_batch + b * M * K;
        float* C = C_batch + b * M * N;
        
        // Prefetch B into cache
        for (int k = 0; k < K; k += PREFETCH_STRIDE) {
            PREFETCH_READ(B + k * N);
        }
        
#if IS_ARM_PLATFORM
        matmul_neon_8x_unroll(A, B, C, M, N, K);
#elif IS_X86_PLATFORM
        matmul_avx2_8x_unroll(A, B, C, M, N, K);
#else
        matmul_blocked(A, B, C, M, N, K);
#endif
    }
}

// 3. Vectorized LayerNorm with SIMD Reduction
#if IS_X86_PLATFORM
void layernorm_avx2(float* output, const float* input, const float* weight,
                    const float* bias, int size) {
    constexpr int AVX_SIZE = 8;
    
    // Compute mean (vectorized)
    __m256 sum_vec = _mm256_setzero_ps();
    int i = 0;
    for (; i + AVX_SIZE <= size; i += AVX_SIZE) {
        __m256 vals = _mm256_loadu_ps(&input[i]);
        sum_vec = _mm256_add_ps(sum_vec, vals);
    }
    for (; i < size; i++) {
        sum_vec = _mm256_add_ss(sum_vec, _mm256_set1_ps(input[i]));
    }
    
    float mean = _mm256_reduce_add_ps(sum_vec) / size;
    __m256 mean_vec = _mm256_set1_ps(mean);
    
    // Compute variance (vectorized)
    __m256 var_vec = _mm256_setzero_ps();
    i = 0;
    for (; i + AVX_SIZE <= size; i += AVX_SIZE) {
        __m256 vals = _mm256_loadu_ps(&input[i]);
        vals = _mm256_sub_ps(vals, mean_vec);
        vals = _mm256_mul_ps(vals, vals);
        var_vec = _mm256_add_ps(var_vec, vals);
    }
    for (; i < size; i++) {
        float diff = input[i] - mean;
        var_vec = _mm256_add_ss(var_vec, _mm256_set1_ps(diff * diff));
    }
    
    float var = _mm256_reduce_add_ps(var_vec) / size;
    float inv_std = 1.0f / std::sqrt(var + 1e-5f);
    __m256 scale_vec = _mm256_set1_ps(inv_std);
    __m256 w_vec = _mm256_set1_ps(weight ? weight[0] : 1.0f);
    __m256 b_vec = _mm256_set1_ps(bias ? bias[0] : 0.0f);
    
    // Normalize and apply weight/bias
    i = 0;
    for (; i + AVX_SIZE <= size; i += AVX_SIZE) {
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
    float mean = vgetq_lane_f32(sum_vec, 0);
    for (int j = 1; j < 4 && i - 4 + j < size; j++) {
        mean += vgetq_lane_f32(sum_vec, j);
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
    float var = vgetq_lane_f32(var_vec, 0);
    for (int j = 1; j < 4 && i - 4 + j < size; j++) {
        float diff = input[i - 4 + j] - mean;
        var += diff * diff;
    }
    for (; i < size; i++) {
        float diff = input[i] - mean;
        var += diff * diff;
    }
    var /= size;
    
    float inv_std = 1.0f / std::sqrt(var + 1e-5f);
    float32x4_t scale_vec = vdupq_n_f32(inv_std);
    float32x4_t w_vec = vdupq_f32(weight ? weight[0] : 1.0f);
    float32x4_t b_vec = vdupq_f32(bias ? bias[0] : 0.0f);
    
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

// 4. Optimized Attention Score Computation
void attention_scores_optimized(const float* Q, const float* K, float* scores,
                                int B, int T, int d, float scale) {
    const int head_dim = d;
    
    for (int b = 0; b < B; b++) {
        const float* Q_head = Q + b * T * head_dim;
        const float* K_head = K + b * T * head_dim;
        float* scores_row = scores + b * T * T;
        
        for (int qi = 0; qi < T; qi++) {
            const float* Q_vec = Q_head + qi * head_dim;
            float* score_out = scores_row + qi * T;
            
            // Vectorized dot product
#if IS_X86_PLATFORM
            constexpr int AVX_SIZE = 8;
            __m256 sum_vec = _mm256_setzero_ps();
            int h = 0;
            for (; h + AVX_SIZE <= head_dim; h += AVX_SIZE) {
                __m256 qv = _mm256_loadu_ps(&Q_vec[h]);
                __m256 kv = _mm256_loadu_ps(&K_head[h]);
                sum_vec = _mm256_fmadd_ps(qv, kv, sum_vec);
            }
            float sum = _mm256_reduce_add_ps(sum_vec);
            for (; h < head_dim; h++) {
                sum += Q_vec[h] * K_head[h];
            }
            score_out[0] = sum * scale;
#elif IS_ARM_PLATFORM
            constexpr int NEON_SIZE = 4;
            float32x4_t sum_vec = vdupq_f32(0.0f);
            int h = 0;
            for (; h + NEON_SIZE <= head_dim; h += NEON_SIZE) {
                float32x4_t qv = vld1q_f32(&Q_vec[h]);
                float32x4_t kv = vld1q_f32(&K_head[h]);
                sum_vec = vfmaq_f32(sum_vec, qv, kv);
            }
            float sum = vgetq_lane_f32(sum_vec, 0);
            for (int j = 1; j < 4 && h + j < head_dim; j++) {
                sum += vgetq_lane_f32(sum_vec, j);
            }
            for (; h < head_dim; h++) {
                sum += Q_vec[h] * K_head[h];
            }
            score_out[0] = sum * scale;
#else
            float sum = 0.0f;
            for (int h = 0; h < head_dim; h++) {
                sum += Q_vec[h] * K_head[h];
            }
            score_out[0] = sum * scale;
#endif
        }
    }
}

// 5. Parallel Batch Processing with OpenMP
#ifdef _OPENMP
void matmul_batch_parallel(const float* A_batch, const float* B, float* C_batch,
                           int batch_size, int M, int N, int K) {
    #pragma omp parallel for schedule(dynamic, 1)
    for (int b = 0; b < batch_size; b++) {
#if IS_ARM_PLATFORM
        matmul_neon_8x_unroll(A_batch + b * M * K, B, C_batch + b * M * N, M, N, K);
#elif IS_X86_PLATFORM
        matmul_avx2_8x_unroll(A_batch + b * M * K, B, C_batch + b * M * N, M, N, K);
#else
        matmul_naive(A_batch + b * M * K, B, C_batch + b * M * N, M, N, K);
#endif
    }
}
#endif // _OPENMP

// ==================== Session 52 Optimization Complete ====================

// ==================== Quantized Matrix Multiplication ====================
HOT_FUNC inline unsigned char quantize(float x) {
    return x > 0.0f ? 1 : 0;
}

// LUT for popcount optimization
static const uint8_t POPCOUNT_LUT[256] = {
    0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,
    1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
    1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
    2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
    1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
    2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
    2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
    3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8
};

HOT_FUNC inline int fast_popcount(uint8_t x) {
    return POPCOUNT_LUT[x];
}

int popcount_bytes(const unsigned char* data, int len) {
    int count = 0;
    int i = 0;
    
    // Process 8 bytes at a time for better efficiency
    for (; i + 7 < len; i += 8) {
        uint64_t val;
        std::memcpy(&val, data + i, sizeof(val));
        count += __builtin_popcountll(val);
    }
    
    // Handle remainder
    for (; i < len; i++) {
        count += POPCOUNT_LUT[data[i]];
    }
    
    return count;
}

// 1-bit matrix multiplication using popcount
void quantized_matmul(const BitMatrix& A, const BitMatrix& B, float* C,
                      int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int matches = 0;
            
            // XOR and count matching bits (1-bit dot product)
            for (int k = 0; k < K; k += 8) {
                int chunk = std::min(8, K - k);
                unsigned char a_val = (A.data[i * A.stride_bytes + k / 8] >> (k % 8)) & 0xFF;
                unsigned char b_val = (B.data[j * B.stride_bytes + k / 8] >> (k % 8)) & 0xFF;
                unsigned char xored = a_val ^ b_val;
                matches += POPCOUNT_LUT[xored];
            }
            
            // Convert to bipolar: matching = +1, mismatching = -1
            C[i * N + j] = 2.0f * matches - chunk;
        }
    }
}
