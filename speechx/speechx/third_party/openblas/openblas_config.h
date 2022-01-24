#ifndef OPENBLAS_CONFIG_H
#define OPENBLAS_CONFIG_H
#define OPENBLAS_OS_LINUX 1
#define OPENBLAS_ARCH_X86_64 1
#define OPENBLAS_C_GCC 1
#define OPENBLAS___64BIT__ 1
#define OPENBLAS_PTHREAD_CREATE_FUNC pthread_create
#define OPENBLAS_BUNDERSCORE _
#define OPENBLAS_NEEDBUNDERSCORE 1
#define OPENBLAS_HASWELL 
#define OPENBLAS_L1_CODE_SIZE 32768
#define OPENBLAS_L1_CODE_ASSOCIATIVE 8
#define OPENBLAS_L1_CODE_LINESIZE 64
#define OPENBLAS_L1_DATA_SIZE 32768
#define OPENBLAS_L1_DATA_ASSOCIATIVE 8
#define OPENBLAS_L1_DATA_LINESIZE 64
#define OPENBLAS_L2_SIZE 262144
#define OPENBLAS_L2_ASSOCIATIVE 8
#define OPENBLAS_L2_LINESIZE 64
#define OPENBLAS_ITB_SIZE 2097152
#define OPENBLAS_ITB_ASSOCIATIVE 0
#define OPENBLAS_ITB_ENTRIES 8
#define OPENBLAS_DTB_SIZE 4096
#define OPENBLAS_DTB_ASSOCIATIVE 4
#define OPENBLAS_DTB_DEFAULT_ENTRIES 64
#define OPENBLAS_HAVE_CMOV 
#define OPENBLAS_HAVE_MMX 
#define OPENBLAS_HAVE_SSE 
#define OPENBLAS_HAVE_SSE2 
#define OPENBLAS_HAVE_SSE3 
#define OPENBLAS_HAVE_SSSE3 
#define OPENBLAS_HAVE_SSE4_1 
#define OPENBLAS_HAVE_SSE4_2 
#define OPENBLAS_HAVE_AVX 
#define OPENBLAS_HAVE_AVX2 
#define OPENBLAS_HAVE_FMA3 
#define OPENBLAS_HAVE_CFLUSH 
#define OPENBLAS_NUM_SHAREDCACHE 2
#define OPENBLAS_NUM_CORES 8
#define OPENBLAS_CORE_HASWELL 
#define OPENBLAS_CHAR_CORENAME "HASWELL"
#define OPENBLAS_SLOCAL_BUFFER_SIZE 24576
#define OPENBLAS_DLOCAL_BUFFER_SIZE 32768
#define OPENBLAS_CLOCAL_BUFFER_SIZE 12288
#define OPENBLAS_ZLOCAL_BUFFER_SIZE 8192
#define OPENBLAS_GEMM_MULTITHREAD_THRESHOLD 4
#define OPENBLAS_VERSION " OpenBLAS 0.3.7 "
/*This is only for "make install" target.*/

#if defined(OPENBLAS_OS_WINNT) || defined(OPENBLAS_OS_CYGWIN_NT) || defined(OPENBLAS_OS_INTERIX)
#define OPENBLAS_WINDOWS_ABI
#define OPENBLAS_OS_WINDOWS

#ifdef DOUBLE
#define DOUBLE_DEFINED DOUBLE
#undef  DOUBLE
#endif
#endif

#ifdef OPENBLAS_NEEDBUNDERSCORE
#define BLASFUNC(FUNC) FUNC##_
#else
#define BLASFUNC(FUNC) FUNC
#endif

#ifdef OPENBLAS_QUAD_PRECISION
typedef struct {
  unsigned long x[2];
}  xdouble;
#elif defined OPENBLAS_EXPRECISION
#define xdouble long double
#else
#define xdouble double
#endif

#if defined(OPENBLAS_OS_WINDOWS) && defined(OPENBLAS___64BIT__)
typedef long long BLASLONG;
typedef unsigned long long BLASULONG;
#else
typedef long BLASLONG;
typedef unsigned long BLASULONG;
#endif

#ifdef OPENBLAS_USE64BITINT
typedef BLASLONG blasint;
#else
typedef int blasint;
#endif

#if defined(XDOUBLE) || defined(DOUBLE)
#define FLOATRET	FLOAT
#else
#ifdef NEED_F2CCONV
#define FLOATRET	double
#else
#define FLOATRET	float
#endif
#endif

/* Inclusion of a standard header file is needed for definition of __STDC_*
   predefined macros with some compilers (e.g. GCC 4.7 on Linux).  This occurs
   as a side effect of including either <features.h> or <stdc-predef.h>. */
#include <stdio.h>

/* C99 supports complex floating numbers natively, which GCC also offers as an
   extension since version 3.0.  If neither are available, use a compatible
   structure as fallback (see Clause 6.2.5.13 of the C99 standard). */
#if ((defined(__STDC_IEC_559_COMPLEX__) || __STDC_VERSION__ >= 199901L || \
      (__GNUC__ >= 3 && !defined(__cplusplus))) && !(defined(FORCE_OPENBLAS_COMPLEX_STRUCT))) && !defined(_MSC_VER)
  #define OPENBLAS_COMPLEX_C99
#ifndef __cplusplus
  #include <complex.h>
#endif
  typedef float _Complex openblas_complex_float;
  typedef double _Complex openblas_complex_double;
  typedef xdouble _Complex openblas_complex_xdouble;
  #define openblas_make_complex_float(real, imag)    ((real) + ((imag) * _Complex_I))
  #define openblas_make_complex_double(real, imag)   ((real) + ((imag) * _Complex_I))
  #define openblas_make_complex_xdouble(real, imag)  ((real) + ((imag) * _Complex_I))
  #define openblas_complex_float_real(z)             (creal(z))
  #define openblas_complex_float_imag(z)             (cimag(z))
  #define openblas_complex_double_real(z)            (creal(z))
  #define openblas_complex_double_imag(z)            (cimag(z))
  #define openblas_complex_xdouble_real(z)           (creal(z))
  #define openblas_complex_xdouble_imag(z)           (cimag(z))
#else
  #define OPENBLAS_COMPLEX_STRUCT
  typedef struct { float real, imag; } openblas_complex_float;
  typedef struct { double real, imag; } openblas_complex_double;
  typedef struct { xdouble real, imag; } openblas_complex_xdouble;
  #define openblas_make_complex_float(real, imag)    {(real), (imag)}
  #define openblas_make_complex_double(real, imag)   {(real), (imag)}
  #define openblas_make_complex_xdouble(real, imag)  {(real), (imag)}
  #define openblas_complex_float_real(z)             ((z).real)
  #define openblas_complex_float_imag(z)             ((z).imag)
  #define openblas_complex_double_real(z)            ((z).real)
  #define openblas_complex_double_imag(z)            ((z).imag)
  #define openblas_complex_xdouble_real(z)           ((z).real)
  #define openblas_complex_xdouble_imag(z)           ((z).imag)
#endif
#endif /* OPENBLAS_CONFIG_H */
