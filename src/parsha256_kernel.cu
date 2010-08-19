#include "common.h"
#include "parsha256.h"
#include <stdio.h>

#define ch_256(x, y, z) ((x & y) ^ (~x & z))
#define maj_256(x, y, z) ((x & y) ^ (x & z) ^ (y & z))
#define Sigma0_256(x) (ROTATER(x, 2) ^ ROTATER(x, 13) ^ ROTATER(x, 22))
#define Sigma1_256(x) (ROTATER(x, 6) ^ ROTATER(x, 11) ^ ROTATER(x, 25))
#define sigma0_256(x) (ROTATER(x, 7) ^ ROTATER(x, 18) ^ SHIFTR(x, 3))
#define sigma1_256(x) (ROTATER(x, 17) ^ ROTATER(x, 19) ^ SHIFTR(x, 10))


/*
 * Table of round constants.
 * First 32 bits of the fractional parts of the cube roots of the first 64 primes 2..311
 */
__device__ static const unsigned int K256[] = {
	0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
	0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
	0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
	0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
	0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
	0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
	0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
	0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};


/*
 * Process one block
 */
__device__ void sha256 (unsigned char *input, unsigned char *output)
{
	unsigned long W[64], a, b, c, d, e, f, g, h;
	unsigned long a1, b1, c1, d1, e1, f1, g1, h1;
	unsigned long t1, t2;
	int t;

	for (t = 0; t < 16; t++)
		/* Add 32 because first 8 words are intermediate hash state */
		GET_UINT32_BE(W[t], input, t * 4 + 32);
	for (; t < 64; t++)
		W[t] = sigma1_256(W[t - 2]) + W[t - 7] + sigma0_256(W[t - 15]) + W[t - 16];

	/* intermediate hash state */
	GET_UINT32_BE(a, input,  0);
	GET_UINT32_BE(b, input,  4);
	GET_UINT32_BE(c, input,  8);
	GET_UINT32_BE(d, input, 12);
	GET_UINT32_BE(e, input, 16);
	GET_UINT32_BE(f, input, 20);
	GET_UINT32_BE(g, input, 24);
	GET_UINT32_BE(h, input, 28);

	a1 = a;
	b1 = b;
	c1 = c;
	d1 = d;
	e1 = e;
	f1 = f;
	g1 = g;
	h1 = h;

	for (t = 0; t < 64; t++) {
		t1 = h + Sigma1_256(e) + ch_256(e, f, g) + K256[t] + W[t];
		t2 = Sigma0_256(a) + maj_256(a, b, c);
		h = g;
		g = f;
		f = e;
		e = d + t1;
		d = c;
		c = b;
		b = a;
		a = t1 + t2;
	}

	a = a + a1;
	b = b + b1;
	c = c + c1;
	d = d + d1;
	e = e + e1;
	f = f + f1;
	g = g + g1;
	h = h + h1;

	PUT_UINT32_BE(a, output, 0);
	PUT_UINT32_BE(b, output, 4);
	PUT_UINT32_BE(c, output, 8);
	PUT_UINT32_BE(d, output, 12);
	PUT_UINT32_BE(e, output, 16);
	PUT_UINT32_BE(f, output, 20);
	PUT_UINT32_BE(g, output, 24);
	PUT_UINT32_BE(h, output, 28);
}


__global__ void parsha256_kernel (unsigned char *input, unsigned char *output, unsigned long total_threads)
{
	unsigned long thread_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (thread_index > total_threads - 1)
		return;

	sha256(&input[thread_index * 96], &output[thread_index * 32]);
}
