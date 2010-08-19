/*
 * SHA-1 GPU implementation.
 * 2008, Tadas Vilkeliskis <vilkeliskis.t@gmail.com>
 */
#include <cuda.h>
#include "common.h"


#define S(x,n) ((x << n) | ((x & 0xFFFFFFFF) >> (32 - n)))
#define R(t) \
	temp = extended[block_index + t -  3] ^ extended[block_index + t - 8] ^     \
		   extended[block_index + t - 14] ^ extended[block_index + t - 16]; \
	extended[block_index + t] = S(temp,1); \


typedef struct {
	unsigned long state[5];
} sha1_gpu_context;

/*
 * Process extended block.
 */
__device__ void sha1_gpu_process (sha1_gpu_context *ctx, unsigned long W[80])
{
	unsigned long A, B, C, D, E;
	A = ctx->state[0];
	B = ctx->state[1];
	C = ctx->state[2];
	D = ctx->state[3];
	E = ctx->state[4];

#define P(a,b,c,d,e,x)                                  \
{                                                       \
    e += S(a,5) + F(b,c,d) + K + x; b = S(b,30);        \
}


#define F(x,y,z) (z ^ (x & (y ^ z)))
#define K 0x5A827999
  
	P( A, B, C, D, E, W[0]  );
	P( E, A, B, C, D, W[1]  );
	P( D, E, A, B, C, W[2]  );
	P( C, D, E, A, B, W[3]  );
	P( B, C, D, E, A, W[4]  );
	P( A, B, C, D, E, W[5]  );
	P( E, A, B, C, D, W[6]  );
	P( D, E, A, B, C, W[7]  );
	P( C, D, E, A, B, W[8]  );
	P( B, C, D, E, A, W[9]  );
	P( A, B, C, D, E, W[10] );
	P( E, A, B, C, D, W[11] );
	P( D, E, A, B, C, W[12] );
	P( C, D, E, A, B, W[13] );
	P( B, C, D, E, A, W[14] );
	P( A, B, C, D, E, W[15] );
	P( E, A, B, C, D, W[16] );
	P( D, E, A, B, C, W[17] );
	P( C, D, E, A, B, W[18] );
	P( B, C, D, E, A, W[19] );
  
#undef K
#undef F

#define F(x,y,z) (x ^ y ^ z)
#define K 0x6ED9EBA1
  
	P( A, B, C, D, E, W[20] );
	P( E, A, B, C, D, W[21] );
	P( D, E, A, B, C, W[22] );
	P( C, D, E, A, B, W[23] );
	P( B, C, D, E, A, W[24] );
	P( A, B, C, D, E, W[25] );
	P( E, A, B, C, D, W[26] );
	P( D, E, A, B, C, W[27] );
	P( C, D, E, A, B, W[28] );
	P( B, C, D, E, A, W[29] );
	P( A, B, C, D, E, W[30] );
	P( E, A, B, C, D, W[31] );
	P( D, E, A, B, C, W[32] );
	P( C, D, E, A, B, W[33] );
	P( B, C, D, E, A, W[34] );
	P( A, B, C, D, E, W[35] );
	P( E, A, B, C, D, W[36] );
	P( D, E, A, B, C, W[37] );
	P( C, D, E, A, B, W[38] );
	P( B, C, D, E, A, W[39] );
  
#undef K
#undef F

#define F(x,y,z) ((x & y) | (z & (x | y)))
#define K 0x8F1BBCDC

	P( A, B, C, D, E, W[40] );
	P( E, A, B, C, D, W[41] );
	P( D, E, A, B, C, W[42] );
	P( C, D, E, A, B, W[43] );
	P( B, C, D, E, A, W[44] );
	P( A, B, C, D, E, W[45] );
	P( E, A, B, C, D, W[46] );
	P( D, E, A, B, C, W[47] );
	P( C, D, E, A, B, W[48] );
	P( B, C, D, E, A, W[49] );
	P( A, B, C, D, E, W[50] );
	P( E, A, B, C, D, W[51] );
	P( D, E, A, B, C, W[52] );
	P( C, D, E, A, B, W[53] );
	P( B, C, D, E, A, W[54] );
	P( A, B, C, D, E, W[55] );
	P( E, A, B, C, D, W[56] );
	P( D, E, A, B, C, W[57] );
	P( C, D, E, A, B, W[58] );
	P( B, C, D, E, A, W[59] );
  
#undef K
#undef F

#define F(x,y,z) (x ^ y ^ z)
#define K 0xCA62C1D6
  
	P( A, B, C, D, E, W[60] );
	P( E, A, B, C, D, W[61] );
	P( D, E, A, B, C, W[62] );
	P( C, D, E, A, B, W[63] );
	P( B, C, D, E, A, W[64] );
	P( A, B, C, D, E, W[65] );
	P( E, A, B, C, D, W[66] );
	P( D, E, A, B, C, W[67] );
	P( C, D, E, A, B, W[68] );
	P( B, C, D, E, A, W[69] );
	P( A, B, C, D, E, W[70] );
	P( E, A, B, C, D, W[71] );
	P( D, E, A, B, C, W[72] );
	P( C, D, E, A, B, W[73] );
	P( B, C, D, E, A, W[74] );
	P( A, B, C, D, E, W[75] );
	P( E, A, B, C, D, W[76] );
	P( D, E, A, B, C, W[77] );
	P( C, D, E, A, B, W[78] );
	P( B, C, D, E, A, W[79] );
  
#undef K
#undef F

	ctx->state[0] += A;
	ctx->state[1] += B;
	ctx->state[2] += C;
	ctx->state[3] += D;
	ctx->state[4] += E;
}

__global__ void sha1_kernel_global (unsigned char *data, sha1_gpu_context *ctx, int total_threads, unsigned long *extended)
{
	int thread_index = threadIdx.x + blockDim.x * blockIdx.x;
	int e_index = thread_index * 80;
	int block_index = thread_index * 64;
	unsigned long temp, t;

	if (thread_index > total_threads -1)
		return;

	/*
	 * Extend 32 block byte block into 80 byte block.
	 */
	GET_UINT32_BE( extended[e_index    ], data + block_index,  0 );
	GET_UINT32_BE( extended[e_index + 1], data + block_index,  4 );
	GET_UINT32_BE( extended[e_index + 2], data + block_index,  8 );
	GET_UINT32_BE( extended[e_index + 3], data + block_index, 12 );
	GET_UINT32_BE( extended[e_index + 4], data + block_index, 16 );
	GET_UINT32_BE( extended[e_index + 5], data + block_index, 20 );
	GET_UINT32_BE( extended[e_index + 6], data + block_index, 24 );
	GET_UINT32_BE( extended[e_index + 7], data + block_index, 28 );
	GET_UINT32_BE( extended[e_index + 8], data + block_index, 32 );
	GET_UINT32_BE( extended[e_index + 9], data + block_index, 36 );
	GET_UINT32_BE( extended[e_index +10], data + block_index, 40 );
	GET_UINT32_BE( extended[e_index +11], data + block_index, 44 );
	GET_UINT32_BE( extended[e_index +12], data + block_index, 48 );
	GET_UINT32_BE( extended[e_index +13], data + block_index, 52 );
	GET_UINT32_BE( extended[e_index +14], data + block_index, 56 );
	GET_UINT32_BE( extended[e_index +15], data + block_index, 60 );

	for (t = 16; t < 80; t++) {
			temp = extended[e_index + t - 3] ^ extended[e_index + t - 8] ^
				extended[e_index + t - 14] ^ extended[e_index + t - 16];
			extended[e_index + t] = S(temp,1);
	}

	/* Wait for the last thread and compute intermediate hash values of extended blocks */
	__syncthreads();
	if (thread_index == total_threads - 1) {
		for (t = 0; t < total_threads; t++)
			sha1_gpu_process (ctx, (unsigned long*)&extended[t * 80]);
	}
}

