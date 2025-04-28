extern "C" __global__
void super_fast_kernel(const float* u, const float* v, float* results, int size) {{
    // Shared memory for parallel reduction
    __shared__ float sdata[512];

    // Initialize accumulators for UU, VV, UV
    float acc_uu = 0;
    float acc_vv = 0;
    float acc_uv = 0;

    // Grid stride loop for processing large arrays
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < size; 
         i += blockDim.x * gridDim.x) {{
        float u_val = u[i];
        float v_val = v[i];

        // Compute products and accumulate in thread's registers
        acc_uu += u_val * u_val;
        acc_vv += v_val * v_val;
        acc_uv += u_val * v_val;
    }}

    // First level of reduction in shared memory
    int tid = threadIdx.x;
    sdata[tid] = acc_uu;
    __syncthreads();

    // Parallel reduction for UU
    for (int s = blockDim.x/2; s > 0; s >>= 1) {{
        if (tid < s) {{
            sdata[tid] += sdata[tid + s];
        }}
        __syncthreads();
    }}

    // Thread 0 writes UU result
    if (tid == 0) atomicAdd(&results[0], sdata[0]);

    // Repeat for VV
    sdata[tid] = acc_vv;
    __syncthreads();

    for (int s = blockDim.x/2; s > 0; s >>= 1) {{
        if (tid < s) {{
            sdata[tid] += sdata[tid + s];
        }}
        __syncthreads();
    }}

    if (tid == 0) atomicAdd(&results[1], sdata[0]);

    // Repeat for UV
    sdata[tid] = acc_uv;
    __syncthreads();

    for (int s = blockDim.x/2; s > 0; s >>= 1) {{
        if (tid < s) {{
            sdata[tid] += sdata[tid + s];
        }}
        __syncthreads();
    }}

    if (tid == 0) atomicAdd(&results[2], sdata[0]);
}}
