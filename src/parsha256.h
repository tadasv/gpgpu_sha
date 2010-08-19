#ifndef __PARSHA256_H__
#define __PARSHA256_H__

/* 2 to the power of a */
#define POW2(a) ((unsigned)1 << (a))
#define DELTA(i) (POW2(i) * (2 * PARSHA256_BLOCK_SIZE - 2 * PARSHA256_HASH_SIZE - PARSHA256_IV_SIZE) - (PARSHA256_BLOCK_SIZE - 2 * PARSHA256_HASH_SIZE))
#define LAMDA(i) (POW2(i -1 ) * (2 * PARSHA256_BLOCK_SIZE - 2 * PARSHA256_HASH_SIZE - PARSHA256_IV_SIZE))
/* Hash function domain in bits */
#define PARSHA256_BLOCK_SIZE	768
/* Hash function range in bits */
#define PARSHA256_HASH_SIZE	256
/* Length of IV in bits */
#define PARSHA256_IV_SIZE	256
/* Available processor tree */
#define TREE_SIZE		16

#define PARSHA256_256BITSB 32
#define PARSHA256_512BITSB 64
#define PARSHA256_768BITSB 96

#endif /* __PARSHA256_H__ */
