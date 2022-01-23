// modify form kaldi-types.h

#ifndef BASE_BASIC_TYPES_H_
#define BASE_BASIC_TYPES_H_ 1


#include <stdint.h>
#include <stddef.h>
#include "base/macros.h"

// for discussion on what to do if you need compile kaldi
// without OpenFST, see the bottom of this this file
/*#include <fst/types.h>

namespace kaldi {
  using ::int16;
  using ::int32;
  using ::int64;
  using ::uint16;
  using ::uint32;
  using ::uint64;
  typedef float   float32;
  typedef double double64;
}  // end namespace kaldi
*/


namespace goat {
  typedef signed char   int8;
  typedef short  int16;
  typedef int  int32;

  typedef unsigned char  uint8;
  typedef unsigned short uint16;
  typedef unsigned int uint32;

	typedef float float32;
}  // end namespace goat 


#endif  // BASE_BASIC_TYPES_H_
