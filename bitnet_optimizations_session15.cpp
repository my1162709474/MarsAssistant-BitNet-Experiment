/**
 * BitNet Session 15: Additional Advanced Optimizations
 * 
 * Optimizations:
 * - Sparse Matrix Multiplication (CSR format)
 * - Mixed Precision Training Support
 * - Aggressive 16x Loop Unrolling
 * - Memory Pool Allocator
 * - OpenMP Dynamic Scheduling
 * - Tile-Based Matrix Multiplication
 * - Exponential Moving Average (EMA)
 * - Layer Normalization Vectorized
 * - GELU Activation Approximation
 * - Softmax Vectorized
 * 
 * Author: MarsAssistant
 * Date: 2026-02-01
 */

#include <cmath>
#include <cstring>
#include <vector>
#include <atomic>
#include <algorithm>
#include <immintrin.h>
#include <arm_neon.h>

// ============================================================================
// 1. MEMORY POOL ALLOCATOR
// ============================================================================
class MemoryPool {
private:
    std::vector<void*> free_blocks;
    std::vector<void*> used_blocks;
    size_t block_size;
    size_t num_blocks;
    std::atomic<size_t> allocation_count{0};
    
public:
    MemoryPool(size_t block_size_bytes, size_t num_blocks_hint = 64)
        : block_size(block_size_bytes), num_blocks(num_blocks_hint) {
        // Pre-allocate blocks aligned to 64-byte cache lines
        for (size_t i = 0; i < num_blocks; i++) {
            void* ptr = nullptr;
            posix_memalign(&ptr, 64, block_size);
            if (ptr) {
                free_blocks.push_back(ptr);
            }
        }
    }
    
    ~MemoryPool() {
        // Free all blocks
        for (void* ptr : free_blocks) free(ptr);
        for (void* ptr : used_blocks) free(ptr);
    }
    
    void* allocate() {
        std::lock_guard<std::mutex> lock(mutex);
        if (!free_blocks.empty()) {
            void* ptr = free_blocks.back();
            free_blocks.pop_back();
            used_blocks.push_back(ptr);
            allocation_count++;
            return ptr;
        }
        // Fallback to regular allocation
        void* ptr = nullptr;
        posix_memalign(&ptr, 64, block_size);
        if (ptr) {
            used_blocks.push_back(ptr);
            allocation_count++;
        }
        return ptr;
    }
    
    void deallocate(void* ptr) {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = std::find(used_blocks.begin(), used_blocks.end(), ptr);
        if (it != used_blocks.end()) {
            used_blocks.erase(it);
            free_blocks.push_back(ptr);
        }
    }
    
    size_t get_allocation_count() const { return allocation_count.load(); }
    
private:
    std::mutex mutex;
};

// Global memory pool for tensor allocations
static MemoryPool* global_tensor_pool = nullptr;

// ============================================================================
// 2. SPARSE MATRIX MULTIPLICATION (CSR FORMAT)
// ============================================================================
struct CSRMatrix {
    int* row_ptr;      // Row pointers (size: m+1)
    int* col_idx;      // Column indices (size: nnz)
    float* values;     // Non-zero values (size: nnz)
    int rows, cols, nnz;
};

// Sparse matrix-vector multiplication (AVX2)
void sparse_matvec_avx2(const CSRMatrix& A, const float* x, float* y) {
    // Initialize output
    std::fill_n(y, A.rows, 0.0f);
    
    for (int i = 0; i < A.rows; i++) {
        float sum = 0.0f;
        int row_start = A.row_ptr[i];
        int row_end = A.row_ptr[i + 1];
        
        // Process 8 elements at a time with AVX2
        int j = row_start;
        for (; j + 7 < row_end; j += 8) {
            __m256 x_vec = _mm256_loadu_ps(&x[A.col_idx[j]]);
            __m256 a_vec = _mm256_loadu_ps(&A.values[j]);
            __m256 prod = _mm256_mul_ps(x_vec, a_vec);
            __m256 sum_vec = _mm256_set1_ps(sum);
            sum_vec = _mm256_add_ps(sum_vec, prod);
            sum = _mm256_cvtss_f32(_mm256_hadd_ps(sum_vec, sum_vec));
        }
        
        // Handle remaining elements
        for (; j < row_end; j++) {
            sum += A.values[j] * x[A.col_idx[j]];
        }
        
        y[i] = sum;
    }
}

// ============================================================================
// 3. MIXED PRECISION TRAINING
// ============================================================================
struct MixedPrecisionState {
    float* master_weights;     // FP32 master weights
    float* fp16_weights;       // FP16 working weights
    float loss_scale;
    bool use_mixed_precision;
    
    MixedPrecisionState(int size, float initial_scale = 1.0f) {
        master_weights = new float[size]();
        fp16_weights = new float[size]();
        loss_scale = initial_scale;
        use_mixed_precision = false;
    }
    
    ~MixedPrecisionState() {
        delete[] master_weights;
        delete[] fp16_weights;
    }
    
    // Convert FP32 to BF16 with rounding
    void fp32_to_bf16(const float* src, uint16_t* dst, int n) {
        for (int i = 0; i < n; i++) {
            dst[i] = (uint16_t)(std::round(src[i]));
        }
    }
    
    // Dynamic loss scaling
    void update_loss_scale(bool overflow_detected) {
        if (overflow_detected) {
            loss_scale = std::max(1.0f, loss_scale / 2.0f);
        } else {
            // Gradual increase if no overflow for many steps
            loss_scale = std::min(32768.0f, loss_scale * 1.01f);
        }
    }
};

// ============================================================================
// 4. TILE-BASED MATRIX MULTIPLICATION (64x64 tiles)
// ============================================================================
#define TILE_SIZE 64

void matmul_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
    constexpr int TS = TILE_SIZE;
    
    for (int i = 0; i < M; i += TS) {
        for (int j = 0; j < N; j += TS) {
            // Initialize tile
            for (int ii = 0; ii < TS && i + ii < M; ii++) {
                for (int jj = 0; jj < TS && j + jj < N; jj++) {
                    C[(i + ii) * N + (j + jj)] = 0.0f;
                }
            }
            
            // Multiply tiles
            for (int k = 0; k < K; k += TS) {
                for (int ii = 0; ii < TS && i + ii < M; ii++) {
                    for (int kk = 0; kk < TS && k + kk < K; kk++) {
                        float a_val = A[(i + ii) * K + (k + kk)];
                        for (int jj = 0; jj < TS && j + jj < N; jj++) {
                            C[(i + ii) * N + (j + jj)] += a_val * B[(k + kk) * N + (j + jj)];
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// 5. VECTORIZED LAYER NORMALIZATION (NEON)
// ============================================================================
void layernorm_neon(const float* x, float* y, float* mean, float* var, 
                    int n, float eps = 1e-5f) {
    // Compute mean
    float sum = 0.0f;
    int i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t v = vld1q_f32(&x[i]);
        sum += vaddvq_f32(v);
    }
    for (; i < n; i++) sum += x[i];
    *mean = sum / n;
    
    // Compute variance
    float var_sum = 0.0f;
    float mean_val = *mean;
    i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t v = vld1q_f32(&x[i]);
        float32x4_t m = vdupq_n_f32(mean_val);
        float32x4_t diff = vsubq_f32(v, m);
        var_sum += vaddvq_f32(vmulq_f32(diff, diff));
    }
    for (; i < n; i++) {
        float diff = x[i] - mean_val;
        var_sum += diff * diff;
    }
    *var = var_sum / n;
    
    // Compute output: (x - mean) / sqrt(var + eps)
    float std_inv = 1.0f / std::sqrt(*var + eps);
    i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t v = vld1q_f32(&x[i]);
        float32x4_t m = vdupq_n_f32(mean_val);
        float32x4_t s = vdupq_n_f32(std_inv);
        v = vsubq_f32(v, m);
        v = vmulq_f32(v, s);
        vst1q_f32(&y[i], v);
    }
    for (; i < n; i++) {
        y[i] = (x[i] - mean_val) * std_inv;
    }
}

// ============================================================================
// 6. GELU ACTIVATION APPROXIMATION (NEON)
// ============================================================================
float gelu_fast_approx(float x) {
    // Fast approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    float c = 0.044715f;
    float sqrt_2_over_pi = 0.79788456f;
    return 0.5f * x * (1.0f + std::tanh(sqrt_2_over_pi * (x + c * x * x * x)));
}

void gelu_fast_neon(const float* x, float* y, int n) {
    float c = 0.044715f;
    float sqrt_2_over_pi = 0.79788456f;
    
    int i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t v = vld1q_f32(&x[i]);
        float32x4_t c4 = vdupq_n_f32(c);
        float32x4_t s4 = vdupq_n_f32(sqrt_2_over_pi);
        float32x4_t half = vdupq_n_f32(0.5f);
        float32x4_t one = vdupq_n_f32(1.0f);
        
        // x^3
        float32x4_t x3 = vmulq_f32(v, vmulq_f32(v, v));
        // x + 0.044715 * x^3
        float32x4_t t = vaddq_f32(v, vmulq_f32(c4, x3));
        // tanh(sqrt(2/pi) * t)
        float32x4_t th = vtanhq_f32(vmulq_f32(s4, t));
        // 1 + tanh
        th = vaddq_f32(one, th);
        // 0.5 * x * (1 + tanh)
        v = vmulq_f32(vmulq_f32(half, v), th);
        vst1q_f32(&y[i], v);
    }
    for (; i < n; i++) {
        y[i] = gelu_fast_approx(x[i]);
    }
}

// ============================================================================
// 7. VECTORIZED SOFTMAX (NEON)
// ============================================================================
void softmax_neon(const float* x, float* y, int n) {
    // Find max for numerical stability
    float max_val = x[0];
    int i = 1;
    for (; i + 3 < n; i += 4) {
        float32x4_t v = vld1q_f32(&x[i]);
        float32x4_t m = vdupq_n_f32(max_val);
        max_val = std::max(max_val, vmaxvq_f32(v));
    }
    for (; i < n; i++) {
        max_val = std::max(max_val, x[i]);
    }
    
    // Compute exp(x - max) and sum
    float sum = 0.0f;
    i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t v = vld1q_f32(&x[i]);
        float32x4_t m = vdupq_n_f32(max_val);
        v = vexpq_f32(vsubq_f32(v, m));  // exp(v - max)
        vst1q_f32(&y[i], v);
        sum += vaddvq_f32(v);
    }
    for (; i < n; i++) {
        y[i] = std::exp(x[i] - max_val);
        sum += y[i];
    }
    
    // Normalize
    float inv_sum = 1.0f / sum;
    i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t v = vld1q_f32(&y[i]);
        float32x4_t s = vdupq_n_f32(in_sum);
        v = vmulq_f32(v, s);
        vst1q_f32(&y[i], v);
    }
    for (; i < n; i++) {
        y[i] *= inv_sum;
    }
}

// ============================================================================
// 8. OPENMP DYNAMIC SCHEDULING
// ============================================================================
void matmul_omp_dynamic(const float* A, const float* B, float* C,
                        int M, int N, int K, int chunk_size = 256) {
    #pragma omp parallel for schedule(dynamic, chunk_size)
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            float a_val = A[i * K + k];
            if (a_val == 0.0f) continue;  // Skip zeros
            for (int j = 0; j < N; j++) {
                C[i * N + j] += a_val * B[k * N + j];
            }
        }
    }
}

// ============================================================================
// 9. EXPONENTIAL MOVING AVERAGE (EMA)
// ============================================================================
class EMA {
private:
    float decay;
    float* cached_weights;
    bool initialized;
    
public:
    EMA(float decay_rate = 0.999f) : decay(decay_rate), initialized(false) {
        cached_weights = nullptr;
    }
    
    ~EMA() {
        delete[] cached_weights;
    }
    
    void update(float* weights, int size) {
        if (!initialized) {
            cached_weights = new float[size];
            std::copy(weights, weights + size, cached_weights);
            initialized = true;
        } else {
            float one_minus_decay = 1.0f - decay;
            for (int i = 0; i < size; i++) {
                cached_weights[i] = decay * cached_weights[i] + one_minus_decay * weights[i];
                weights[i] = cached_weights[i];  // In-place update
            }
        }
    }
    
    // Fast inference mode (no momentum)
    void inference_copy(float* weights, int size) {
        if (initialized) {
            std::copy(cached_weights, cached_weights + size, weights);
        }
    }
};

// ============================================================================
// 10. AGGRESSIVE 16x LOOP UNROLLING
// ============================================================================
void matmul_unrolled_16x(const float* A, const float* B, float* C,
                         int M, int N, int K) {
    constexpr int UNROLL = 16;
    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j += UNROLL) {
            __m256 c_vec[UNROLL/2];
            for (int u = 0; u < UNROLL/2; u++) {
                c_vec[u] = _mm256_setzero_ps();
            }
            
            for (int k = 0; k < K; k++) {
                __m256 a_val = _mm256_set1_ps(A[i * K + k]);
                for (int u = 0; u < UNROLL/2; u++) {
                    if (j + u*2 + 1 < N) {
                        __m256 b_vec = _mm256_loadu_ps(&B[k * N + j + u*2]);
                        c_vec[u] = _mm256_fmadd_ps(a_val, b_vec, c_vec[u]);
                    }
                }
            }
            
            for (int u = 0; u < UNROLL/2; u++) {
                if (j + u*2 + 1 < N) {
                    _mm256_storeu_ps(&C[i * N + j + u*2], c_vec[u]);
                }
            }
        }
    }
}

