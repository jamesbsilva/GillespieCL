/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement [EULA] associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#define   MT_RNG_COUNT 4096
#define   MT_MM 9
#define   MT_NN 19
#define   MT_WMASK 0xFFFFFFFFU
#define   MT_UMASK 0xFFFFFFFEU
#define   MT_UMASK 0xFFFFFFFEU

#define      MT_LMASK 0x1U
#define      MATRIX_A 0x9908b0df /* Constant vector a */
#define      MT_SHIFT0 12
#define      MT_SHIFTB 7
#define	     MASK_B 0x9d2c5680
#define      MT_SHIFTC 15
#define      MASK_C 0xefc60000
#define      MT_SHIFT1 18
#define PI 3.14159265358979f

////////////////////////////////////////////////////////////////////////////////
// OpenCL Kernel for Mersenne Twister RNG [Modified from NVIDIA Example]
////////////////////////////////////////////////////////////////////////////////
__kernel void mersenne_twister(__global float* d_Rand,
			      unsigned int  seed, unsigned int seed2 ,
			      int nPerRng)
{
    int globalID = get_global_id(0);

    int iState, iState1, iStateM, iOut;
    unsigned int mti, mti1, mtiM, x;
    unsigned int mt[MT_NN]; 

    //Initialize current state
    mt[0] = seed+((globalID+seed2)%globalID);
    for (iState = 1; iState < MT_NN; iState++)
        mt[iState] = (1812433253U * (mt[iState - 1] ^ (mt[iState - 1] >> 30)) + iState) & MT_WMASK;

    iState = 0;
    mti1 = mt[0];
    for (iOut = 0; iOut < (nPerRng+2); iOut++) {
        iState1 = iState + 1;
        iStateM = iState + MT_MM;
        if(iState1 >= MT_NN) iState1 -= MT_NN;
        if(iStateM >= MT_NN) iStateM -= MT_NN;
        mti  = mti1;
        mti1 = mt[iState1];
        mtiM = mt[iStateM];

        // MT recurrence
        x = (mti & MT_UMASK) | (mti1 & MT_LMASK);
	    x = mtiM ^ (x >> 1) ^ ((x & 1) ? MATRIX_A : 0);

        mt[iState] = x;
        iState = iState1;

        //Tempering transformation
        x ^= (x >> MT_SHIFT0);
        x ^= (x << MT_SHIFTB) & MASK_B;
        x ^= (x << MT_SHIFTC) & MASK_C;
        x ^= (x >> MT_SHIFT1);

        //Convert to (0, 1] float and write to global memory
        d_Rand[globalID] = ((float)x + 1.0f) / 4294967296.0f;
     
    }

}
