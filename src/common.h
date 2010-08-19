#ifndef __COMMON_H__
#define __COMMON_H__

/*
 * 32-bit integer manipulation macros (big endian)
 */
#ifndef GET_UINT32_BE
#define GET_UINT32_BE(n,b,i)\
{\
    (n) = ( (unsigned long) (b)[(i) ] << 24 )\
        | ( (unsigned long) (b)[(i) + 1] << 16 )\
        | ( (unsigned long) (b)[(i) + 2] <<  8 )\
        | ( (unsigned long) (b)[(i) + 3]       );\
}
#endif

#ifndef RETURN_UINT32_BE
#define RETURN_UINT32_BE(b,i)\
(\
    	( (unsigned long) (b)[(i) ] << 24 )\
    	| ( (unsigned long) (b)[(i) + 1] << 16 )\
        | ( (unsigned long) (b)[(i) + 2] <<  8 )\
        | ( (unsigned long) (b)[(i) + 3]       )\
)
#endif


#ifndef GET_UINT32_BE_GPU
#define GET_UINT32_BE_GPU(n,b,i)\
{\
    (n) = ( (unsigned long) (b)[(i) + 3] << 24 )\
        | ( (unsigned long) (b)[(i) + 2] << 16 )\
        | ( (unsigned long) (b)[(i) + 1] <<  8 )\
        | ( (unsigned long) (b)[(i) ]       );\
}
#endif


#ifndef PUT_UINT32_BE
#define PUT_UINT32_BE(n,b,i)\
{\
    (b)[(i)    ] = (unsigned char) ( (n) >> 24 );	\
    (b)[(i) + 1] = (unsigned char) ( (n) >> 16 );	\
    (b)[(i) + 2] = (unsigned char) ( (n) >>  8 );	\
    (b)[(i) + 3] = (unsigned char) ( (n)       );	\
}
#endif


#define	TRUNCLONG(x)	(x)
/* Circular rotation to the right for 32 bit word */
#define	ROTATER(x,n)	(((x) >> (n)) | ((x) << (32 - (n))))
/* Shift to the right */
#define	SHIFTR(x,n)		((x) >> (n))

/* Little-Endian to Big-Endian for 32 bit word */
#define LETOBE32(i) (((i) & 0xff) << 24) + (((i) & 0xff00) << 8) + (((i) & 0xff0000) >> 8) + (((i) >> 24) & 0xff)
/* Return number of 0 bytes to pad */
#define padding_256(len)	(((len) & 0x3f) < 56) ? (56 - ((len) & 0x3f)) : (120 - ((len) & 0x3f))


#endif /* __COMMON_H__ */

