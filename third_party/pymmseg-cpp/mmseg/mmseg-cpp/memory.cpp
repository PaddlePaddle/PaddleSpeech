#include "memory.h"

#define PRE_ALLOC_SIZE 2097152 /* 2MB */

namespace rmmseg {
char *_pool_base = static_cast<char *>(std::malloc(PRE_ALLOC_SIZE));
size_t _pool_size = PRE_ALLOC_SIZE;
}
