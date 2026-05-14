#pragma once

#if defined(__has_include_next)
#if __has_include_next(<math.h>)
#include_next <math.h>
#endif
#else
#include_next <math.h>
#endif

#define isnan(x) __isnan(x)
