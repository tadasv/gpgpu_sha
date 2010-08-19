/*
 * PARSHA-256 benchmark program. Calculates execution time of PARSHA-256 on CPU and GPU.
 * Also includes function parsha256_gpu which prepares PARSHA-256 to executes on GPU and
 * executes it.
 *
 * 2008, Tadas Vilkeliskis <vilkeliskis.t@gmail.com>
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cutil.h>
#include "parsha256.h"

typedef struct {
	unsigned int kernel_timer;	/* execution time of the kernel */
	unsigned int malloc_timer;	/* time spent on memory allocation */
	unsigned int memcpy_timer;	/* time spent on copying memory from hsot to device and vise versa */
	unsigned int free_timer;	/* time spent on memory deallocation */
} chronometer;

chronometer chmeter = {0, 0, 0, 0};

extern __global__ void parsha256_kernel (unsigned char *input, unsigned char *output, unsigned long total_threads);

void parsha256_gpu (unsigned char *input, unsigned long size, unsigned char *output)
{
	unsigned long t;		/* effective tree height */
	unsigned char *d_input;		/* input buffer on device */
	unsigned char *d_output;	/* intermediate hash states */
	int total_threads;		/* Total number of threads in the grid */
	int threads_per_block = 128;	/* Maximum number of threads per block */
	int total_blocks;		/* Total blocks in the grid */
	unsigned char *buffer_ptr;	/* Pointer to input buffer */
	unsigned long bytes_read = 0;	/* Bytes read from the input */
	unsigned long q, r, b, s, k;
	int l1, K1, L1;
	/*
	 * Initialization vector. Length 256 bits. Since reference machine is using 64 bit words
	 * char array was used instead of word array. I was experiencing some problems with words.
	 */
	const unsigned char IV[32] =    {0x67, 0xe6, 0x09, 0x6a,
					0x85, 0xae, 0x67, 0xbb,
					0x72, 0xf3, 0x6e, 0x3c,
					0x3a, 0xf5, 0x4f, 0xa5,
					0x7f, 0x52, 0x0e, 0x51,
					0x8c, 0x68, 0x05, 0x9b,
					0xab, 0xd9, 0x83, 0x1f,
					0x19, 0xcd, 0xe0, 0x5b};
	/* Few temporary variables */
	int i, j;
	unsigned long tmp1, tmp2;

	size = size * 8; /* bytes to bits */

	if (size <= 160 * 8) {
		/*
		 * if L <= delta0 = n - l, then return h(h(x||0^(n-l-L)||IV)||bin_(n-m)(L))
		 * */
		printf ("Not implemented for size less than %d bits\n", 160 * 8);
		return;
	}

	/* BEGIN INITIALIZATION */
	/* Determine effective tree height */
	if (size >= DELTA(TREE_SIZE))
		t = TREE_SIZE;
	else {
		for (i = TREE_SIZE - 1; i >= 1; i--)
			if (DELTA(i) <= size && size < DELTA(i + 1)) {
				t = i;
				i = 0; /* break the loop */
			}
	}

	/* Find other parameters needed to complete computation */
	q = r = 0;
	if (size > DELTA(t)) {
		q = (size - DELTA(t)) / LAMDA(t);
		r = (size - DELTA(t)) % LAMDA(t);
		if (r == 0) {
			q--;
			r = LAMDA(t);
		}
	}

	b = r / (2 * PARSHA256_BLOCK_SIZE - 2 * PARSHA256_HASH_SIZE - PARSHA256_IV_SIZE);
	if (r % (2 * PARSHA256_BLOCK_SIZE - 2 * PARSHA256_HASH_SIZE - PARSHA256_IV_SIZE))
		b++;

	/* Total number of processors for the first round */
	total_threads = POW2(t);
#if 0
#ifdef _DEBUG
	printf ("tree size: %d\n", t);
	printf ("total threads: %d\n", total_threads);
	printf ("q, r, b: %d %d %d\n", q, r, b);
#endif
#endif
	CUT_SAFE_CALL (cutResetTimer (chmeter.malloc_timer));
	CUT_SAFE_CALL (cutStartTimer (chmeter.malloc_timer));
	/* Allocate enough memory on the device */
	cudaMalloc ((void**)&d_input, total_threads * PARSHA256_768BITSB);
	CUT_CHECK_ERROR ("Memory allocation failed");
	cudaMalloc ((void**)&d_output, total_threads * PARSHA256_256BITSB);
	CUT_CHECK_ERROR ("Memory allocation failed");
	CUT_SAFE_CALL (cutStopTimer (chmeter.malloc_timer));

	/* END INITIALIZATION */

	/* BEGIN FIRST ROUND */
	buffer_ptr = input;
	CUT_SAFE_CALL (cutResetTimer (chmeter.memcpy_timer));
	CUT_SAFE_CALL (cutStartTimer (chmeter.memcpy_timer));
	for (i = 0; i < total_threads; i++) {
		/* Copy 512 bits */
		cudaMemcpy(d_input + i * PARSHA256_768BITSB, buffer_ptr, PARSHA256_512BITSB,
				cudaMemcpyHostToDevice);
		CUT_CHECK_ERROR ("Memory copy failed");
		/* Add 256 bits of IV */
		cudaMemcpy(d_input + i * PARSHA256_768BITSB + PARSHA256_512BITSB,
				(unsigned char *)&IV, PARSHA256_256BITSB, cudaMemcpyHostToDevice);
		CUT_CHECK_ERROR ("Memory copy failed");
		buffer_ptr += PARSHA256_512BITSB;
		bytes_read += PARSHA256_512BITSB;
	}
	CUT_SAFE_CALL (cutStopTimer (chmeter.memcpy_timer));

	/* execute kernel */
	total_blocks = total_threads / threads_per_block + (total_threads % threads_per_block == 0 ? 0 : 1);
#if 0
#ifdef _DEBUG
	printf ("bytes read: %d\n", bytes_read);
	printf ("total blocks: %d\n", total_blocks);
	printf ("total_threads: %d\n", total_threads);
	printf ("threads_per_block: %d\n", threads_per_block);
#endif
#endif
	CUT_SAFE_CALL (cutResetTimer (chmeter.kernel_timer));
	CUT_SAFE_CALL (cutStartTimer (chmeter.kernel_timer));
	parsha256_kernel <<<total_blocks, threads_per_block>>> (d_input, d_output, total_threads);
	CUT_CHECK_ERROR ("Kernel execution failed");
	CUT_SAFE_CALL (cutStopTimer (chmeter.kernel_timer));

	/* END FIRST ROUND */
	/* BEGIN STEADY STATE */
	tmp2 = q + 1;
	for (i = 2; i <= tmp2; i++) {
		tmp1 = POW2 (t - 1) - 1;

		CUT_SAFE_CALL (cutStartTimer (chmeter.memcpy_timer));

		for (j = 0; j <= tmp1; j++) {
			/* Copy intermediate hash states */
			cudaMemcpy (d_input + j * PARSHA256_768BITSB, d_output + j * PARSHA256_512BITSB,
					PARSHA256_256BITSB, cudaMemcpyDeviceToDevice);
			CUT_CHECK_ERROR ("Memory copy failed");
			cudaMemcpy (d_input + j * PARSHA256_768BITSB + PARSHA256_256BITSB,
					d_output + j * PARSHA256_512BITSB + PARSHA256_256BITSB,
					PARSHA256_256BITSB,
					cudaMemcpyDeviceToDevice);
			CUT_CHECK_ERROR ("Memory copy failed");
			/* Copy 256 bits from input message */
			cudaMemcpy (d_input + j * PARSHA256_768BITSB + PARSHA256_512BITSB, buffer_ptr,
					PARSHA256_256BITSB, cudaMemcpyHostToDevice);
			buffer_ptr += PARSHA256_256BITSB;
			bytes_read += PARSHA256_256BITSB;
		}

		tmp1 = POW2 (t) - 1;
		for (j = POW2 (t - 1); j <= tmp1; j++) {
			/* Copy 512 bits */
			cudaMemcpy(d_input + j * PARSHA256_768BITSB, buffer_ptr, PARSHA256_512BITSB,
					cudaMemcpyHostToDevice);
			CUT_CHECK_ERROR ("Memory copy failed");
			/* Add 256 bits of IV */
			cudaMemcpy(d_input + j * PARSHA256_768BITSB + PARSHA256_512BITSB,
					(unsigned char *)&IV, PARSHA256_256BITSB, cudaMemcpyHostToDevice);
			CUT_CHECK_ERROR ("Memory copy failed");
			buffer_ptr += PARSHA256_512BITSB;
			bytes_read += PARSHA256_512BITSB;
		}

		CUT_SAFE_CALL (cutStopTimer (chmeter.memcpy_timer));

		/* execute kernel */
		total_blocks = total_threads / threads_per_block + (total_threads % threads_per_block == 0 ? 0 : 1);
#if 0
#ifdef _DEBUG
		printf ("bytes read (steady state): %d\n", bytes_read);
		printf ("total blocks: %d\n", total_blocks);
		printf ("total_threads: %d\n", total_threads);
		printf ("threads_per_block: %d\n", threads_per_block);
#endif
#endif
		CUT_SAFE_CALL (cutStartTimer (chmeter.kernel_timer));
		parsha256_kernel <<<total_blocks, threads_per_block>>> (d_input, d_output, total_threads);
		CUT_CHECK_ERROR ("Kernel execution failed");
		CUT_SAFE_CALL (cutStopTimer (chmeter.kernel_timer));
	}

	tmp1 = POW2(t - 1) - 1;
	total_threads = POW2(t - 1) + b - 1;

	CUT_SAFE_CALL (cutStartTimer (chmeter.memcpy_timer));

	for (i = 0; i <= tmp1; i++) {
		/* Copy intermediate hash states */
		cudaMemcpy (d_input + i * PARSHA256_768BITSB, d_output + i * PARSHA256_512BITSB,
				PARSHA256_256BITSB, cudaMemcpyDeviceToDevice);
		CUT_CHECK_ERROR ("Memory copy failed");
		cudaMemcpy (d_input + i * PARSHA256_768BITSB + PARSHA256_256BITSB,
				d_output + i * PARSHA256_512BITSB + PARSHA256_256BITSB,
				PARSHA256_256BITSB,
				cudaMemcpyDeviceToDevice);
		CUT_CHECK_ERROR ("Memory copy failed");
		/* Copy 256 bits from input message */
		cudaMemcpy (d_input + i * PARSHA256_768BITSB + PARSHA256_512BITSB, buffer_ptr,
				PARSHA256_256BITSB, cudaMemcpyHostToDevice);
		buffer_ptr += PARSHA256_256BITSB;
		bytes_read += PARSHA256_256BITSB;
	}

	for (i = POW2(t - 1); i <= total_threads; i++) {
		/* Copy 512 bits */
		cudaMemcpy(d_input + i * PARSHA256_768BITSB, buffer_ptr, PARSHA256_512BITSB,
				cudaMemcpyHostToDevice);
		CUT_CHECK_ERROR ("Memory copy failed");
		/* Add 256 bits of IV */
		cudaMemcpy(d_input + i * PARSHA256_768BITSB + PARSHA256_512BITSB,
				(unsigned char *)&IV, PARSHA256_256BITSB, cudaMemcpyHostToDevice);
		CUT_CHECK_ERROR ("Memory copy failed");
		buffer_ptr += PARSHA256_512BITSB;
		bytes_read += PARSHA256_512BITSB;
	}

	CUT_SAFE_CALL (cutStopTimer (chmeter.memcpy_timer));

	/* execute kernel */
	total_blocks = total_threads / threads_per_block + (total_threads % threads_per_block == 0 ? 0 : 1);
#if 0
#ifdef _DEBUG
	printf ("bytes read (end game): %d\n", bytes_read);
	printf ("total blocks: %d\n", total_blocks);
	printf ("total_threads: %d\n", total_threads);
	printf ("threads_per_block: %d\n", threads_per_block);
#endif
#endif
	CUT_SAFE_CALL (cutStartTimer (chmeter.kernel_timer));
	parsha256_kernel <<<total_blocks, threads_per_block>>> (d_input, d_output, total_threads);
	CUT_CHECK_ERROR ("Kernel execution failed");
	CUT_SAFE_CALL (cutStopTimer (chmeter.kernel_timer));

	/* BEGIN FLUSHING */
	tmp1 = q + t + 1;
	size = size / 8;	/* back to bytes */
	for (i = q + 3; i <= tmp1; i++) {
		s = q + t + 2 - i;
		k = (b - 1 + POW2 (t - s - 1)) / POW2 (t - s);
		l1 = (b - 1 + POW2 (t - s)) / POW2 (t - s);
		K1 = POW2 (s - 1) + k;
		L1 = POW2 (s - 1) + l1;
		
		/* zero out the buffer for padding I guess */
		CUT_SAFE_CALL (cutStartTimer (chmeter.memcpy_timer));
		cudaMemset(d_input, 0, K1 * PARSHA256_256BITSB);
		CUT_SAFE_CALL (cutStopTimer (chmeter.memcpy_timer));
		tmp2 = K1 - 1;

		if (size - bytes_read >= K1 * PARSHA256_256BITSB)
			bytes_read += (K1 * PARSHA256_256BITSB);
		else
			bytes_read += (size - bytes_read);

		CUT_SAFE_CALL (cutStartTimer (chmeter.memcpy_timer));
		for (j = 0; j <= tmp2; j++) {
			/* Copy intermediate hash states */
			cudaMemcpy (d_input + j * PARSHA256_768BITSB, d_output + j * PARSHA256_512BITSB,
					PARSHA256_256BITSB, cudaMemcpyDeviceToDevice);
			CUT_CHECK_ERROR ("Memory copy failed");
			cudaMemcpy (d_input + j * PARSHA256_768BITSB + PARSHA256_256BITSB,
					d_output + j * PARSHA256_512BITSB + PARSHA256_256BITSB,
					PARSHA256_256BITSB,
					cudaMemcpyDeviceToDevice);
			CUT_CHECK_ERROR ("Memory copy failed");
			/* Copy 256 bits from input message */
			cudaMemcpy (d_input + j * PARSHA256_768BITSB + PARSHA256_512BITSB, buffer_ptr,
					PARSHA256_256BITSB, cudaMemcpyHostToDevice);
			buffer_ptr += PARSHA256_256BITSB;
		}
		CUT_SAFE_CALL (cutStopTimer (chmeter.memcpy_timer));

		/* execute the kernel */
		total_threads = K1;
		total_blocks = total_threads / threads_per_block + (total_threads % threads_per_block == 0 ? 0 : 1);
#if 0
#ifdef _DEBUG
		printf ("bytes readi (flushing): %d\n", bytes_read);
		printf ("total blocks: %d\n", total_blocks);
		printf ("total_threads: %d\n", total_threads);
		printf ("threads_per_block: %d\n", threads_per_block);
#endif
#endif
		CUT_SAFE_CALL (cutStartTimer (chmeter.kernel_timer));
		parsha256_kernel <<<total_blocks, threads_per_block>>> (d_input, d_output, total_threads);
		CUT_CHECK_ERROR ("Kernel execution failed");
		CUT_SAFE_CALL (cutStopTimer (chmeter.kernel_timer));

		tmp2 = L1 - 1;
		CUT_SAFE_CALL (cutStartTimer (chmeter.memcpy_timer));
		for (j = K1; j <= tmp2; j++) {
			cudaMemcpy (d_output + j * PARSHA256_256BITSB, d_output + j * PARSHA256_512BITSB,
					PARSHA256_256BITSB, cudaMemcpyDeviceToDevice);
		}
		CUT_SAFE_CALL (cutStopTimer (chmeter.memcpy_timer));
	}

	total_blocks  = 1;
	total_threads = 1;
	if (b > 0) {
		CUT_SAFE_CALL (cutStartTimer (chmeter.memcpy_timer));
		cudaMemset (d_input, 0, PARSHA256_768BITSB);
		/* Copy intermediate hash states */
		cudaMemcpy (d_input, d_output, PARSHA256_256BITSB, cudaMemcpyDeviceToDevice);
		CUT_CHECK_ERROR ("Memory copy failed");
		cudaMemcpy (d_input + PARSHA256_256BITSB, d_output + PARSHA256_256BITSB,
			PARSHA256_256BITSB, cudaMemcpyDeviceToDevice);
		CUT_CHECK_ERROR ("Memory copy failed");
		CUT_SAFE_CALL (cutStopTimer (chmeter.memcpy_timer));

		if (size - bytes_read >= PARSHA256_256BITSB) {
			CUT_SAFE_CALL (cutStartTimer (chmeter.memcpy_timer));
			cudaMemcpy (d_input + PARSHA256_512BITSB, buffer_ptr, PARSHA256_256BITSB,
					cudaMemcpyHostToDevice);
			CUT_SAFE_CALL (cutStopTimer (chmeter.memcpy_timer));
			buffer_ptr += PARSHA256_256BITSB;
			bytes_read += PARSHA256_256BITSB;
		} else {
			CUT_SAFE_CALL (cutStartTimer (chmeter.memcpy_timer));
			cudaMemcpy (d_input + PARSHA256_512BITSB, buffer_ptr, size - bytes_read,
					cudaMemcpyHostToDevice);
			CUT_SAFE_CALL (cutStopTimer (chmeter.memcpy_timer));
			bytes_read += (size - bytes_read);
		}

		CUT_SAFE_CALL (cutStartTimer (chmeter.kernel_timer));
		parsha256_kernel <<<total_blocks, threads_per_block>>> (d_input, d_output, total_threads);
		CUT_CHECK_ERROR ("Kernel execution failed");
//		cudaThreadSynchronize();
		CUT_SAFE_CALL (cutStopTimer (chmeter.kernel_timer));
	}

	CUT_SAFE_CALL (cutStartTimer (chmeter.memcpy_timer));
	cudaMemset (d_output + PARSHA256_256BITSB, 0, PARSHA256_512BITSB - 8);
	cudaMemcpy (d_input, d_output, PARSHA256_768BITSB, cudaMemcpyDeviceToDevice);
	size = size * 8;
	/*
	 * The following line should fail on 32 bit machines. Since reference machine I
	 * am writing this code on uses 64 bit words thus size of int is 8 bytes.
	 */
	cudaMemcpy (d_input + PARSHA256_768BITSB - 8, &size, 8, cudaMemcpyHostToDevice);

	/* Hash one more time */
	CUT_SAFE_CALL (cutStopTimer (chmeter.memcpy_timer));
	CUT_SAFE_CALL (cutStartTimer (chmeter.kernel_timer));
	parsha256_kernel <<<total_blocks, threads_per_block>>> (d_input, d_output, 1);
	CUT_SAFE_CALL (cutStopTimer (chmeter.kernel_timer));

	/* And we are done here */
	cudaMemcpy (output, d_output, PARSHA256_256BITSB, cudaMemcpyDeviceToHost);

	CUT_SAFE_CALL (cutResetTimer (chmeter.free_timer));
	CUT_SAFE_CALL (cutStartTimer (chmeter.free_timer));
	cudaFree (d_input);
	cudaFree (d_output);
	CUT_SAFE_CALL (cutStopTimer (chmeter.free_timer));
}

int main (int argc, char **argv)
{
	unsigned char *buffer;
	unsigned int size;
	unsigned char output[32];

	printf ("========================================\n");
	printf ("PARSHA-256 HASH ALGORITHM BENCHMARK TEST\n");
	printf ("========================================\n\n");

	CUT_SAFE_CALL (cutCreateTimer ((unsigned int*)&chmeter.kernel_timer));
	CUT_SAFE_CALL (cutCreateTimer ((unsigned int*)&chmeter.malloc_timer));
	CUT_SAFE_CALL (cutCreateTimer ((unsigned int*)&chmeter.memcpy_timer));
	CUT_SAFE_CALL (cutCreateTimer ((unsigned int*)&chmeter.free_timer));

	printf ("SIZE      EXEC KERNEL\tcudaMemcpy\tcudaMalloc\tcudaFree\n");

	for (size = 1000; size <= 100000000; size *= 10) {
		buffer = (unsigned char *) malloc (size * sizeof (char));
		if (buffer == NULL) {
			printf ("Memory allocation failed\n");
			return -1;
		}

		parsha256_gpu (buffer, size, output);
		printf ("%-10d%f\t%f\t%f\t%f\n", size,
				cutGetTimerValue (chmeter.kernel_timer),
				cutGetTimerValue (chmeter.memcpy_timer),
				cutGetTimerValue (chmeter.malloc_timer),
				cutGetTimerValue (chmeter.free_timer));


		free (buffer);
	}
}
