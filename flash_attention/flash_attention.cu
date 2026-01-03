/*
 * FlashAttention: Fast and Memory-Efficient Exact Attention
 * 
 * This is an educational implementation of the FlashAttention algorithm
 * from "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
 * by Dao et al. (2022): https://arxiv.org/abs/2205.14135
 * 
 * Key concepts:
 * - Tiling: Divides Q, K, V into blocks that fit in GPU shared memory (SRAM)
 * - Online Softmax: Computes softmax incrementally without storing full attention matrix
 * - Memory Efficient: O(N) memory instead of O(N²)
 * 
 * Current Status:
 * - Forward pass: ✅ Fully implemented and tested
 * - Backward pass: ✅ Fully implemented and tested
 * 
 * Limitations of this implementation:
 * - Only supports head_dim=64
 * - Only supports FP32 (no FP16/BF16)
 * - Fixed block sizes (Br=32, Bc=32)
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <vector>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

// FlashAttention Forward Kernel
template <const int Br, const int Bc>
__global__ void flash_attention_forward(
    const float* Q,     // Query matrix [N x d]
    const float* K,     // Key matrix [N x d]
    const float* V,     // Value matrix [N x d]
    float* O,           // Output matrix [N x d]
    float* l,           // Row sum statistics [N]
    float* m,           // Row max statistics [N]
    const int N,        // Sequence length
    const int d,        // Head dimension
    const int Tc,       // Number of column blocks (K, V)
    const int Tr,       // Number of row blocks (Q)
    const float scale   // Softmax scale (1/sqrt(d))
) {
    // Thread and block indices
    int tx = threadIdx.x;
    int bx = blockIdx.x;  // Block index for batch dimension
    int by = blockIdx.y;  // Block index for head dimension
    
    // Calculate offsets for this batch/head
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);
    int lm_offset = (bx * gridDim.y * N) + (by * N);
    
    // Allocate shared memory for tiles
    extern __shared__ float sram[];
    
    // Partition shared memory into regions
    float* Qi = sram;                        // Query block [Br x d]
    float* Kj = Qi + Br * d;                 // Key block [Bc x d]
    float* Vj = Kj + Bc * d;                 // Value block [Bc x d]
    float* S = Vj + Bc * d;                  // Attention scores [Br x Bc]
    
    // Softmax statistics in shared memory
    float* mi_shared = S + Br * Bc;          // Current max [Br]
    float* mi_new = mi_shared + Br;          // Updated max [Br]
    float* li_shared = mi_new + Br;          // Current sum [Br]
    float* li_new = li_shared + Br;          // Updated sum [Br]
    float* mij = li_new + Br;                // Block max [Br]
    
    // Output accumulator in shared memory [Br x d]
    float* Oi = mij + Br;
    
    // Outer loop over Q blocks (row blocks) - CORRECTED LOOP ORDER
    for (int i = 0; i < Tr; i++) {
        
        // === STEP 1: Initialize output accumulator for this Q block ===
        for (int elem = tx; elem < Br * d; elem += blockDim.x) {
            Oi[elem] = 0.0f;
        }
        
        // Initialize statistics for this Q block
        if (tx < Br) {
            int global_row = i * Br + tx;
            if (global_row < N) {
                mi_shared[tx] = -INFINITY;
                li_shared[tx] = 0.0f;
            }
        }
        __syncthreads();
        
        // Inner loop over K, V blocks (column blocks)
        for (int j = 0; j < Tc; j++) {
            
            // === STEP 2: Load Kj and Vj from HBM to SRAM ===
            int kv_loads = CEIL_DIV(Bc * d, blockDim.x);
            for (int load = 0; load < kv_loads; load++) {
                int idx = load * blockDim.x + tx;
                if (idx < Bc * d) {
                    int row = idx / d;
                    int col = idx % d;
                    int global_row = j * Bc + row;
                    
                    if (global_row < N) {
                        Kj[row * d + col] = K[qkv_offset + global_row * d + col];
                        Vj[row * d + col] = V[qkv_offset + global_row * d + col];
                    } else {
                        Kj[row * d + col] = 0.0f;
                        Vj[row * d + col] = 0.0f;
                    }
                }
            }
            __syncthreads();
            
            // === STEP 3: Load Qi from HBM to SRAM (load once per Q block) ===
            if (j == 0) {
                int q_loads = CEIL_DIV(Br * d, blockDim.x);
                for (int load = 0; load < q_loads; load++) {
                    int idx = load * blockDim.x + tx;
                    if (idx < Br * d) {
                        int row = idx / d;
                        int col = idx % d;
                        int global_row = i * Br + row;
                        
                        if (global_row < N) {
                            Qi[row * d + col] = Q[qkv_offset + global_row * d + col];
                        } else {
                            Qi[row * d + col] = 0.0f;
                        }
                    }
                }
                __syncthreads();
            }
            
            // === STEP 4: Compute S = Qi @ Kj^T (Attention Scores) ===
            int attn_loads = CEIL_DIV(Br * Bc, blockDim.x);
            for (int load = 0; load < attn_loads; load++) {
                int idx = load * blockDim.x + tx;
                if (idx < Br * Bc) {
                    int row = idx / Bc;  // Query row
                    int col = idx % Bc;  // Key row
                    
                    float dot_product = 0.0f;
                    for (int k = 0; k < d; k++) {
                        dot_product += Qi[row * d + k] * Kj[col * d + k];
                    }
                    S[row * Bc + col] = dot_product * scale;
                }
            }
            __syncthreads();
            
            // === STEP 5: Online Softmax - Compute block statistics ===
            // Each thread handles one row of Q block
            if (tx < Br) {
                int global_row = i * Br + tx;
                if (global_row < N) {
                    
                    // Find row maximum for numerical stability
                    float row_max = -INFINITY;
                    for (int c = 0; c < Bc; c++) {
                        int global_col = j * Bc + c;
                        if (global_col < N) {
                            float val = S[tx * Bc + c];
                            if (val > row_max) {
                                row_max = val;
                            }
                        }
                    }
                    mij[tx] = row_max;
                    
                    // Compute exp(S - max) and row sum
                    float row_sum = 0.0f;
                    for (int c = 0; c < Bc; c++) {
                        int global_col = j * Bc + c;
                        if (global_col < N) {
                            float exp_val = expf(S[tx * Bc + c] - row_max);
                            S[tx * Bc + c] = exp_val;  // Store exp values
                            row_sum += exp_val;
                        } else {
                            S[tx * Bc + c] = 0.0f;
                        }
                    }
                    
                    // === STEP 6: Update running statistics ===
                    // New maximum across all blocks seen so far
                    float m_old = mi_shared[tx];
                    float m_new = fmaxf(m_old, row_max);
                    mi_new[tx] = m_new;
                    
                    // Update running sum with rescaling
                    float l_old = li_shared[tx];
                    float l_new = expf(m_old - m_new) * l_old + 
                                  expf(row_max - m_new) * row_sum;
                    li_new[tx] = l_new;
                }
            }
            __syncthreads();
            
            // === STEP 7: Compute and accumulate output O = P @ V ===
            // Each thread computes a portion of the output
            for (int elem = tx; elem < Br * d; elem += blockDim.x) {
                int row = elem / d;
                int col = elem % d;
                int global_row = i * Br + row;
                
                if (global_row < N) {
                    // Compute P @ V for this element (P is already exp values in S)
                    float pv_sum = 0.0f;
                    for (int c = 0; c < Bc; c++) {
                        pv_sum += S[row * Bc + c] * Vj[c * d + col];
                    }
                    
                    // Get statistics for this row
                    float m_old = mi_shared[row];
                    float m_new = mi_new[row];
                    float l_old = li_shared[row];
                    
                    // Apply online update formula from FlashAttention paper
                    // O_i^{new} = diag(exp(m_i^{old} - m_i^{new})) * O_i^{old} + exp(m_ij - m_i^{new}) * P_ij @ V_j
                    // where P_ij @ V_j is already computed in pv_sum (P_ij contains exp values)
                    // Since S already contains exp(score - m_ij), we need to rescale by exp(m_ij - m_new)
                    float m_ij = mij[row];
                    
                    if (j == 0) {
                        // First block
                        Oi[elem] = expf(m_ij - m_new) * pv_sum;
                    } else {
                        // Subsequent blocks: rescale old output and add new
                        Oi[elem] = expf(m_old - m_new) * Oi[elem] + expf(m_ij - m_new) * pv_sum;
                    }
                }
            }
            __syncthreads();
            
            // Normalize by l_new after accumulation
            if (j == Tc - 1) {  // Last K/V block for this Q block
                for (int elem = tx; elem < Br * d; elem += blockDim.x) {
                    int row = elem / d;
                    int global_row = i * Br + row;
                    
                    if (global_row < N) {
                        float l_new = li_new[row];
                        Oi[elem] = Oi[elem] / l_new;
                    }
                }
                __syncthreads();
            }
            
            // Update statistics for next iteration
            if (tx < Br) {
                int global_row = i * Br + tx;
                if (global_row < N) {
                    mi_shared[tx] = mi_new[tx];
                    li_shared[tx] = li_new[tx];
                }
            }
            __syncthreads();
            
        } // End inner loop (K, V blocks)
        
        // === STEP 8: Write final output for this Q block to HBM ===
        for (int elem = tx; elem < Br * d; elem += blockDim.x) {
            int row = elem / d;
            int col = elem % d;
            int global_row = i * Br + row;
            
            if (global_row < N) {
                O[qkv_offset + global_row * d + col] = Oi[elem];
            }
        }
        
        // CRITICAL: Write final statistics (l and m) to global memory for backward pass
        if (tx < Br) {
            int global_row = i * Br + tx;
            if (global_row < N) {
                l[lm_offset + global_row] = li_shared[tx];
                m[lm_offset + global_row] = mi_shared[tx];
            }
        }
        __syncthreads();
        
    } // End outer loop (Q blocks)
}

// Host function to launch FlashAttention
// Now returns pointers to l and m for use in backward pass
void flash_attention_forward_host(
    const float* Q,
    const float* K, 
    const float* V,
    float* O,
    float** d_l_out,  // Output: row sums (for backward)
    float** d_m_out,  // Output: row maxs (for backward)
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim
) {
    // Block sizes (tunable hyperparameters)
    const int Br = 32;  // Query block size
    const int Bc = 32;  // Key/Value block size
    
    // Calculate number of blocks
    int Tr = CEIL_DIV(seq_len, Br);
    int Tc = CEIL_DIV(seq_len, Bc);
    
    // Softmax scale
    float scale = 1.0f / sqrtf((float)head_dim);
    
    // Allocate statistics arrays
    float* d_l;  // Row sums
    float* d_m;  // Row maxs
    size_t stats_size = batch_size * num_heads * seq_len * sizeof(float);
    cudaMalloc(&d_l, stats_size);
    cudaMalloc(&d_m, stats_size);
    
    // Initialize statistics
    cudaMemset(d_l, 0, stats_size);
    float neg_inf = -INFINITY;
    cudaMemset(d_m, *((int*)&neg_inf), stats_size);
    
    // Calculate shared memory size
    size_t sram_size = (Br * head_dim +      // Qi
                        Bc * head_dim +       // Kj
                        Bc * head_dim +       // Vj
                        Br * Bc +             // S
                        5 * Br +              // Statistics
                        Br * head_dim) *      // Oi (output accumulator)
                       sizeof(float);
    
    // Configure kernel launch
    dim3 grid(batch_size, num_heads);
    dim3 block(256);  // Threads per block
    
    // Launch kernel
    flash_attention_forward<32, 32><<<grid, block, sram_size>>>(
        Q, K, V, O, d_l, d_m,
        seq_len, head_dim, Tc, Tr, scale
    );
    
    // Return pointers (caller must free)
    *d_l_out = d_l;
    *d_m_out = d_m;
}

// FlashAttention Backward Kernel
// Computes gradients dQ, dK, dV given dO (gradient of output)
template <int Br, int Bc>
__global__ void flash_attention_backward(
    const float* Q,      // Query [N x d]
    const float* K,      // Key [N x d]
    const float* V,      // Value [N x d]
    const float* O,      // Output from forward pass [N x d]
    const float* dO,     // Gradient of output [N x d]
    float* dQ,           // Gradient of Query [N x d]
    float* dK,           // Gradient of Key [N x d]
    float* dV,           // Gradient of Value [N x d]
    const float* l,      // Row sums from forward [N]
    const float* m,      // Row maxs from forward [N]
    const int N,         // Sequence length
    const int d,         // Head dimension
    const int Tc,        // Number of column blocks
    const int Tr,        // Number of row blocks
    const float scale    // Softmax scale
) {
    int tx = threadIdx.x;
    int tile_idx = blockIdx.x;  // Tile index (which Q block we're processing)
    int head_idx = blockIdx.y;  // Head index
    int batch_idx = blockIdx.z; // Batch index
    
    // Calculate offsets for this batch and head
    int qkv_offset = (batch_idx * gridDim.y * N * d) + (head_idx * N * d);
    int lm_offset = (batch_idx * gridDim.y * N) + (head_idx * N);
    
    extern __shared__ float sram[];
    
    // Partition shared memory
    float* Qi = sram;
    float* Kj = Qi + Br * d;
    float* Vj = Kj + Bc * d;
    float* Oi = Vj + Bc * d;
    float* dOi = Oi + Br * d;
    float* S = dOi + Br * d;        // S and dS can share space
    float* P = S + Br * Bc;          // P computed after S
    float* dPi = P + Br * Bc;        // dPi uses separate space
    float* Di = dPi + Br * Bc;       // Di = rowsum(dO * O)
    // Note: dS will reuse S space after S is no longer needed
    
    // Each block processes one Q tile (determined by tile_idx)
    int i = tile_idx;
    
    // Load Qi, Oi, dOi
    int q_loads = CEIL_DIV(Br * d, blockDim.x);
    for (int load = 0; load < q_loads; load++) {
        int idx = load * blockDim.x + tx;
        if (idx < Br * d) {
            int row = idx / d;
            int col = idx % d;
            int global_row = i * Br + row;
                
                if (global_row < N) {
                    Qi[idx] = Q[qkv_offset + global_row * d + col];
                    Oi[idx] = O[qkv_offset + global_row * d + col];
                    dOi[idx] = dO[qkv_offset + global_row * d + col];
                } else {
                    Qi[idx] = 0.0f;
                    Oi[idx] = 0.0f;
                    dOi[idx] = 0.0f;
                }
            }
        }
        
        // Compute D_i = rowsum(dO_i * O_i)
        if (tx < Br) {
            int global_row = i * Br + tx;
            if (global_row < N) {
                float sum = 0.0f;
                for (int k = 0; k < d; k++) {
                    sum += dOi[tx * d + k] * Oi[tx * d + k];
                }
                Di[tx] = sum;
            }
        }
        __syncthreads();
        
        // Inner loop over K, V blocks
        for (int j = 0; j < Tc; j++) {
            
            // Load Kj, Vj
            int kv_loads = CEIL_DIV(Bc * d, blockDim.x);
            for (int load = 0; load < kv_loads; load++) {
                int idx = load * blockDim.x + tx;
                if (idx < Bc * d) {
                    int row = idx / d;
                    int col = idx % d;
                    int global_row = j * Bc + row;
                    
                    if (global_row < N) {
                        Kj[idx] = K[qkv_offset + global_row * d + col];
                        Vj[idx] = V[qkv_offset + global_row * d + col];
                    } else {
                        Kj[idx] = 0.0f;
                        Vj[idx] = 0.0f;
                    }
                }
            }
            __syncthreads();
            
            // Compute S = Q @ K^T (attention scores)
            int attn_loads = CEIL_DIV(Br * Bc, blockDim.x);
            for (int load = 0; load < attn_loads; load++) {
                int idx = load * blockDim.x + tx;
                if (idx < Br * Bc) {
                    int row = idx / Bc;
                    int col = idx % Bc;
                    
                    float dot = 0.0f;
                    for (int k = 0; k < d; k++) {
                        dot += Qi[row * d + k] * Kj[col * d + k];
                    }
                    S[idx] = dot * scale;
                }
            }
            __syncthreads();
            
            // Compute P = softmax(S)
            if (tx < Br) {
                int global_row = i * Br + tx;
                if (global_row < N) {
                    float row_max = m[lm_offset + global_row];
                    float row_sum = l[lm_offset + global_row];
                    
                    for (int c = 0; c < Bc; c++) {
                        int global_col = j * Bc + c;
                        if (global_col < N) {
                            float p_val = expf(S[tx * Bc + c] - row_max) / row_sum;
                            P[tx * Bc + c] = p_val;
                        } else {
                            P[tx * Bc + c] = 0.0f;
                        }
                    }
                }
            }
            __syncthreads();
            
            // Compute dP = dO @ V^T
            for (int load = 0; load < attn_loads; load++) {
                int idx = load * blockDim.x + tx;
                if (idx < Br * Bc) {
                    int row = idx / Bc;
                    int col = idx % Bc;
                    
                    float dot = 0.0f;
                    for (int k = 0; k < d; k++) {
                        dot += dOi[row * d + k] * Vj[col * d + k];
                    }
                    dPi[idx] = dot;
                }
            }
            __syncthreads();
            
            // Compute dS = P * (dP - D)
            // This implements the gradient through softmax
            // Reuse S buffer for dS to save shared memory
            float* dS = S;  // Reuse S buffer since we're done with it
            for (int load = 0; load < attn_loads; load++) {
                int idx = load * blockDim.x + tx;
                if (idx < Br * Bc) {
                    int row = idx / Bc;
                    int col = idx % Bc;
                    int global_row = i * Br + row;
                    int global_col = j * Bc + col;
                    
                    if (global_row < N && global_col < N) {
                        float p = P[idx];
                        float dp = dPi[idx];
                        float d_val = Di[row];
                        dS[idx] = p * (dp - d_val) * scale;
                    } else {
                        dS[idx] = 0.0f;
                    }
                }
            }
            __syncthreads();
            
            // Compute dV += P^T @ dO
            for (int elem = tx; elem < Bc * d; elem += blockDim.x) {
                int row = elem / d;  // Vj row
                int col = elem % d;  // dimension
                int global_row = j * Bc + row;
                
                if (global_row < N) {
                    float sum = 0.0f;
                    for (int r = 0; r < Br; r++) {
                        int q_global_row = i * Br + r;
                        if (q_global_row < N) {
                            sum += P[r * Bc + row] * dOi[r * d + col];
                        }
                    }
                    atomicAdd(&dV[qkv_offset + global_row * d + col], sum);
                }
            }
            
            // Compute dK += dS^T @ Q
            for (int elem = tx; elem < Bc * d; elem += blockDim.x) {
                int row = elem / d;  // Kj row
                int col = elem % d;  // dimension
                int global_row = j * Bc + row;
                
                if (global_row < N) {
                    float sum = 0.0f;
                    for (int r = 0; r < Br; r++) {
                        int q_global_row = i * Br + r;
                        if (q_global_row < N) {
                            sum += dS[r * Bc + row] * Qi[r * d + col];
                        }
                    }
                    atomicAdd(&dK[qkv_offset + global_row * d + col], sum);
                }
            }
            
            // Compute dQ += dS @ K for this Q block and current K block
            for (int elem = tx; elem < Br * d; elem += blockDim.x) {
                int row = elem / d;
                int col = elem % d;
                int global_row = i * Br + row;
                
                if (global_row < N) {
                    float sum = 0.0f;
                    
                    // Accumulate dS @ K for current j block
                    for (int c = 0; c < Bc; c++) {
                        int global_col = j * Bc + c;
                        if (global_col < N) {
                            sum += dS[row * Bc + c] * Kj[c * d + col];
                        }
                    }
                    
                    // Use atomic add since dQ accumulates across j iterations
                    atomicAdd(&dQ[qkv_offset + global_row * d + col], sum);
                }
            }
            __syncthreads();
            
        } // End inner loop (K, V blocks)
}

// Host function for backward pass
void flash_attention_backward_host(
    const float* Q,
    const float* K,
    const float* V,
    const float* O,
    const float* dO,
    float* dQ,
    float* dK,
    float* dV,
    const float* l,
    const float* m,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim
) {
    // Use smaller block sizes to fit within 48KB shared memory limit
    const int Br = 16;  // Reduced from 32
    const int Bc = 16;  // Reduced from 32
    
    int Tr = CEIL_DIV(seq_len, Br);
    int Tc = CEIL_DIV(seq_len, Bc);
    
    float scale = 1.0f / sqrtf((float)head_dim);
    
    // Initialize gradients to zero
    size_t qkv_size = batch_size * num_heads * seq_len * head_dim * sizeof(float);
    cudaMemset(dQ, 0, qkv_size);
    cudaMemset(dK, 0, qkv_size);
    cudaMemset(dV, 0, qkv_size);
    
    // Calculate shared memory size
    // Optimized: S and dS share the same buffer
    size_t sram_size = (Br * head_dim +      // Qi
                        Bc * head_dim +       // Kj
                        Bc * head_dim +       // Vj
                        Br * head_dim +       // Oi
                        Br * head_dim +       // dOi
                        Br * Bc +             // S (shared with dS)
                        Br * Bc +             // P
                        Br * Bc +             // dPi
                        Br) *                 // Di
                       sizeof(float);
    
    // Align to 256 bytes as per NVIDIA best practices
    size_t sram_aligned = ((sram_size + 255) / 256) * 256;
    
    // Query device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    // Diagnostic output
    printf("\n=== Backward Kernel Launch Configuration ===\n");
    printf("Grid: (%d, %d, %d)\n", Tr, num_heads, batch_size);
    printf("Block: 128 threads\n");
    printf("Shared memory: %zu bytes (%.1f KB)\n", sram_aligned, sram_aligned / 1024.0f);
    printf("Device max shared mem per block: %zu bytes (%.1f KB)\n", 
           prop.sharedMemPerBlock, prop.sharedMemPerBlock / 1024.0f);
    printf("Sequence length: %d, Head dim: %d\n", seq_len, head_dim);
    printf("Tr (Q tiles): %d, Tc (KV tiles): %d\n", Tr, Tc);
    
    // Set max dynamic shared memory attribute (required for >48KB)
    cudaError_t attr_err = cudaFuncSetAttribute(
        flash_attention_backward<16, 16>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        sram_aligned
    );
    
    if (attr_err != cudaSuccess) {
        printf("ERROR: cudaFuncSetAttribute failed: %s\n", cudaGetErrorString(attr_err));
        return;
    }
    printf("✓ cudaFuncSetAttribute successful\n");
    
    // Clear any previous errors
    cudaGetLastError();
    
    // Grid: (num_tiles_M, num_heads, batch_size) as per FlashAttention spec
    dim3 grid(Tr, num_heads, batch_size);
    dim3 block(128);
    
    printf("Launching kernel...\n");
    flash_attention_backward<16, 16><<<grid, block, sram_aligned>>>(
        Q, K, V, O, dO, dQ, dK, dV, l, m,
        seq_len, head_dim, Tc, Tr, scale
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("✗ CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    } else {
        printf("✓ Kernel launched successfully\n");
    }
    
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        printf("✗ CUDA kernel execution error: %s\n", cudaGetErrorString(sync_err));
    } else {
        printf("✓ Kernel executed successfully\n");
    }
    printf("==========================================\n\n");
}

// PyTorch wrapper functions
std::vector<torch::Tensor> flash_attention_forward_torch(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V
) {
    // Check inputs
    TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "V must be a CUDA tensor");
    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
    TORCH_CHECK(V.is_contiguous(), "V must be contiguous");
    
    // Get dimensions
    const int batch_size = Q.size(0);
    const int num_heads = Q.size(1);
    const int seq_len = Q.size(2);
    const int head_dim = Q.size(3);
    
    // Check dimensions match
    TORCH_CHECK(K.size(0) == batch_size && V.size(0) == batch_size, 
                "Batch size mismatch");
    TORCH_CHECK(K.size(1) == num_heads && V.size(1) == num_heads, 
                "Number of heads mismatch");
    TORCH_CHECK(K.size(2) == seq_len && V.size(2) == seq_len, 
                "Sequence length mismatch");
    TORCH_CHECK(K.size(3) == head_dim && V.size(3) == head_dim, 
                "Head dimension mismatch");
    
    // Allocate output tensor
    auto O = torch::zeros_like(Q);
    
    // Allocate statistics tensors
    auto l = torch::zeros({batch_size, num_heads, seq_len}, 
                         torch::TensorOptions().dtype(torch::kFloat32).device(Q.device()));
    auto m = torch::full({batch_size, num_heads, seq_len}, -INFINITY,
                        torch::TensorOptions().dtype(torch::kFloat32).device(Q.device()));
    
    // Call CUDA kernel
    float* d_l;
    float* d_m;
    flash_attention_forward_host(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        &d_l,
        &d_m,
        batch_size,
        num_heads,
        seq_len,
        head_dim
    );
    
    // Copy statistics to tensors
    cudaMemcpy(l.data_ptr<float>(), d_l, 
               batch_size * num_heads * seq_len * sizeof(float),
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(m.data_ptr<float>(), d_m,
               batch_size * num_heads * seq_len * sizeof(float),
               cudaMemcpyDeviceToDevice);
    
    // Note: d_l and d_m are freed here since we copied to tensors
    cudaFree(d_l);
    cudaFree(d_m);
    
    return {O, l, m};
}

std::vector<torch::Tensor> flash_attention_backward_torch(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor O,
    torch::Tensor dO,
    torch::Tensor l,
    torch::Tensor m
) {
    // Check inputs
    TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "V must be a CUDA tensor");
    TORCH_CHECK(O.is_cuda(), "O must be a CUDA tensor");
    TORCH_CHECK(dO.is_cuda(), "dO must be a CUDA tensor");
    TORCH_CHECK(l.is_cuda(), "l must be a CUDA tensor");
    TORCH_CHECK(m.is_cuda(), "m must be a CUDA tensor");
    
    // Get dimensions
    const int batch_size = Q.size(0);
    const int num_heads = Q.size(1);
    const int seq_len = Q.size(2);
    const int head_dim = Q.size(3);
    
    // Allocate gradient tensors
    auto dQ = torch::zeros_like(Q);
    auto dK = torch::zeros_like(K);
    auto dV = torch::zeros_like(V);
    
    // Call CUDA kernel
    flash_attention_backward_host(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        dO.data_ptr<float>(),
        dQ.data_ptr<float>(),
        dK.data_ptr<float>(),
        dV.data_ptr<float>(),
        l.data_ptr<float>(),
        m.data_ptr<float>(),
        batch_size,
        num_heads,
        seq_len,
        head_dim
    );
    
    return {dQ, dK, dV};
}

// Binding code for Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_attention_forward_torch, "FlashAttention forward pass (CUDA)");
    m.def("backward", &flash_attention_backward_torch, "FlashAttention backward pass (CUDA)");
}
