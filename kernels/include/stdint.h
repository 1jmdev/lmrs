#pragma once

#if defined(__has_include_next)
#if __has_include_next(<stdint.h>)
#include_next <stdint.h>
#else
typedef signed char int8_t;
typedef short int16_t;
typedef int int32_t;
typedef long long int64_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
#endif
#else
#include_next <stdint.h>
#endif
