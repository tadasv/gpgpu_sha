/*
 * SHA-1 benchmark program. Calculates execution time of SHA-1 on CPU and GPU.
 * Also includes function sha1_gpu_global() which prepares SHA-1 to be executed
 * on GPU.
 *
 * 2008, Tadas Vilkeliskis <vilkeliskis.t@gmail.com>
 */
#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cutil.h>
#include "common.h"

#define MAX_THREADS_PER_BLOCK 128

typedef struct {
	unsigned long state[5];
} sha1_gpu_context;


typedef struct {
	unsigned const char *data;
	unsigned const char *hash;
} testvector;


typedef struct {
	unsigned int kernel_timer;	/* time spent in kernel */
	unsigned int malloc_timer;	/* how much time we spend allocating memory */
	unsigned int memcpy_timer;	/* how much time we spend copying from host to device */
	unsigned int free_timer;	/* how much time we spend releasing memory */
} chronometer;

/* timers used to check performance */
chronometer chmeter = {0, 0, 0, 0};

extern void sha1_cpu (unsigned char *input, int ilen, unsigned char *output);
extern __global__ void sha1_kernel_global (unsigned char *data, sha1_gpu_context *ctx, int total_threads, unsigned long *extended);

/*
 * Run sha1 kernel on GPU
 * input - message
 * size - message size
 * output - buffer to store hash value
 * proc - maximum threads per block
 */
void sha1_gpu_global (unsigned char *input, unsigned long size, unsigned char *output, int proc)
{
	int total_threads;		/* Total number of threads in the grid */
	int blocks_per_grid;		/* Number of blocks in the grid */
	int threads_per_block;		/* Number of threads in a block */
	int pad, size_be;		/* Number of zeros to pad, message size in big-enadian. */
	int total_datablocks;		/* Total number of blocks message is split into */
	int i, k;			/* Temporary variables */
	unsigned char *d_message;	/* Input message on the device */
	unsigned long *d_extended;	/* Extended blocks on the device */
	sha1_gpu_context ctx, *d_ctx;	/* Intermediate hash states */

	/* Initialization vector for SHA-1 */
	ctx.state[0] = 0x67452301;
	ctx.state[1] = 0xEFCDAB89;
	ctx.state[2] = 0x98BADCFE;
	ctx.state[3] = 0x10325476;
	ctx.state[4] = 0xC3D2E1F0;

	pad = padding_256 (size);
	threads_per_block = proc;
	blocks_per_grid = 1;
	/* How many blocks in the message */
	total_datablocks = (size + pad + 8) / 64;

	if (total_datablocks > threads_per_block)
		total_threads = threads_per_block;
	else
		total_threads = total_datablocks;
	
	size_be = LETOBE32 (size * 8);

	/* Allocate enough memory on the device */
	CUT_SAFE_CALL (cutResetTimer (chmeter.malloc_timer));
	CUT_SAFE_CALL (cutStartTimer (chmeter.malloc_timer));
	cudaMalloc ((void**)&d_extended, proc * 80 * sizeof(unsigned long));
	CUT_CHECK_ERROR ("d_extended malloc failed");
	cudaMalloc ((void**)&d_message, size + pad + 8);
	CUT_CHECK_ERROR ("d_message malloc failed");
	cudaMalloc ((void**)&d_ctx, sizeof (sha1_gpu_context));
	CUT_CHECK_ERROR ("d_ctx malloc failed");
	CUT_SAFE_CALL (cutStopTimer (chmeter.malloc_timer));
	CUT_SAFE_CALL (cutResetTimer (chmeter.memcpy_timer));

	/*
	 * Copy the data from host to device and perform padding
	 */
	CUT_SAFE_CALL (cutStartTimer (chmeter.memcpy_timer));
	cudaMemcpy (d_ctx, &ctx, sizeof (sha1_gpu_context), cudaMemcpyHostToDevice);
	cudaMemcpy (d_message, input, size, cudaMemcpyHostToDevice);
	cudaMemset (d_message + size, 0x80, 1);
	cudaMemset (d_message + size + 1, 0, pad + 7);
	cudaMemcpy (d_message + size + pad + 4, &size_be, 4, cudaMemcpyHostToDevice);
	CUT_SAFE_CALL (cutStopTimer (chmeter.memcpy_timer));

	/*
	 * Run the algorithm
	 */
	i = 0;
	k = total_datablocks / total_threads;
	CUT_SAFE_CALL (cutResetTimer (chmeter.kernel_timer));
	CUT_SAFE_CALL (cutStartTimer (chmeter.kernel_timer));
	if (k - 1 > 0) {
		/*
		 * Kernel is executed multiple times and only one block in the grid is used.
		 * Since thread synchronization is allowed only within a block.
		 */
		for (i = 0; i < k; i++) {
			sha1_kernel_global <<<blocks_per_grid, proc>>>(d_message + threads_per_block * i * 64, d_ctx, threads_per_block, d_extended);
			CUT_CHECK_ERROR ("Kernel execution failed");
			/*
			 * Here I do not perform thread synchronization
			 * since threads are shynchronized in the kernel
			 */
		}
	}
	threads_per_block = total_datablocks - (i * total_threads);
	sha1_kernel_global <<<blocks_per_grid, proc>>>(d_message + total_threads * i * 64, d_ctx, threads_per_block, d_extended);
	CUT_CHECK_ERROR ("Kernel execution failed");

	CUT_SAFE_CALL (cutStopTimer (chmeter.kernel_timer));
	CUT_SAFE_CALL (cutStartTimer (chmeter.memcpy_timer));
	cudaMemcpy (&ctx, d_ctx, sizeof(sha1_gpu_context), cudaMemcpyDeviceToHost);
	CUT_SAFE_CALL (cutStopTimer (chmeter.memcpy_timer));

	CUT_SAFE_CALL (cutStartTimer (chmeter.kernel_timer));
	/* Put the hash value in the users' buffer */
	PUT_UINT32_BE( ctx.state[0], output,  0 );
	PUT_UINT32_BE( ctx.state[1], output,  4 );
	PUT_UINT32_BE( ctx.state[2], output,  8 );
	PUT_UINT32_BE( ctx.state[3], output, 12 );
	PUT_UINT32_BE( ctx.state[4], output, 16 );
	CUT_SAFE_CALL (cutStopTimer (chmeter.kernel_timer));

	CUT_SAFE_CALL (cutResetTimer (chmeter.free_timer));
	CUT_SAFE_CALL (cutStartTimer (chmeter.free_timer));
	cudaFree (d_message);
	cudaFree (d_ctx);
	cudaFree (d_extended);
	CUT_SAFE_CALL (cutStopTimer (chmeter.free_timer));
}


int main(int argc, char *argv[])
{
	testvector tv1 = {
		(unsigned char *) "abc",
		(unsigned char *) "\xa9\x99\x3e\x36\x47\x06\x81\x6a\xba\x3e\x25\x71\x78\x50\xc2\x6c\x9c\xd0\xd8\x9d"
	};
	testvector tv2 = {
		(unsigned char *) "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
		(unsigned char *) "\x84\x98\x3e\x44\x1c\x3b\xd2\x6e\xba\xae\x4a\xa1\xf9\x51\x29\xe5\xe5\x46\x70\xf1"
	};
	unsigned char hash[20];
	unsigned char *data = NULL;
	int i;
	int max_threads_per_block = MAX_THREADS_PER_BLOCK;

	printf ("===================================\n");
	printf ("SHA-1 HASH ALGORITHM BENCHMARK TEST\n");
	printf ("===================================\n");

	CUT_SAFE_CALL (cutCreateTimer ((unsigned int*)&chmeter.kernel_timer));
	CUT_SAFE_CALL (cutCreateTimer ((unsigned int*)&chmeter.malloc_timer));
	CUT_SAFE_CALL (cutCreateTimer ((unsigned int*)&chmeter.memcpy_timer));
	CUT_SAFE_CALL (cutCreateTimer ((unsigned int*)&chmeter.free_timer));

	printf ("\nTesting algorithm correctness...\n");

	sha1_cpu ((unsigned char*)tv1.data, strlen((const char*)tv1.data), hash);
	if (memcmp (hash, tv1.hash, 20) == 0) printf ("CPU TEST 1 PASSED\n");
	else printf ("CPU TEST 1 FAILED\n");

	sha1_gpu_global ((unsigned char*)tv1.data, strlen((const char*)tv1.data), hash, max_threads_per_block);
	if (memcmp (hash, tv1.hash, 20) == 0) printf ("GPU TEST 1 PASSED\n");
	else printf ("GPU TEST 1 FAILED\n");
	
	sha1_cpu ((unsigned char*)tv2.data, strlen((const char*)tv2.data), hash);
	if (memcmp (hash, tv2.hash, 20) == 0) printf ("CPU TEST 2 PASSED\n");
	else printf ("CPU TEST 2 FAILED\n");

	sha1_gpu_global ((unsigned char*)tv2.data, strlen((const char*)tv2.data), hash, max_threads_per_block);
	if (memcmp (hash, tv2.hash, 20) == 0) printf ("GPU TEST 2 PASSED\n");
	else printf ("GPU TEST 2 FAILED\n");

	printf ("Done.\n\n");
	printf ("\tSIZE      EXEC KERNEL\tcudaMemcpy\tcudaMalloc\tcudaFree\n");

	for (i = 1000; i < 100000000; i = i * 10) {
		data = (unsigned char *) malloc (i);
		if (data == NULL) {
			printf ("ERROR: Insufficient memory on host\n");
			return -1;
		}

		CUT_SAFE_CALL (cutResetTimer (chmeter.kernel_timer));
		CUT_SAFE_CALL (cutStartTimer (chmeter.kernel_timer));
		sha1_cpu (data, i, hash);
		CUT_SAFE_CALL (cutStopTimer (chmeter.kernel_timer));
		printf ("CPU\t%-10d%f\n", i, cutGetTimerValue (chmeter.kernel_timer));

		CUT_SAFE_CALL (cutResetTimer (chmeter.kernel_timer));
		CUT_SAFE_CALL (cutResetTimer (chmeter.malloc_timer));
		CUT_SAFE_CALL (cutResetTimer (chmeter.memcpy_timer));
		CUT_SAFE_CALL (cutResetTimer (chmeter.free_timer));
		memset (hash, 0, 20);

		sha1_gpu_global (data, i, hash, max_threads_per_block);
		printf ("GPU\t%-10d%f\t%f\t%f\t%f\n", i,
				cutGetTimerValue (chmeter.kernel_timer),
				cutGetTimerValue (chmeter.memcpy_timer),
				cutGetTimerValue (chmeter.malloc_timer),
				cutGetTimerValue (chmeter.free_timer));
		free (data);
	}

	return 0;
}
