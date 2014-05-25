__kernel void fill_float_buffer(__global float *A, const float s0) {
 
    // Get the index of the current element to be processed
    int i = get_global_id(0);
    
    // Do the operation
    A[i] = s0;
}

