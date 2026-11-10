// Catch v1 main. POSIX signal handling is disabled because Catch 1.x uses
// MINSIGSTKSZ as a constant, which is not one on recent glibc.
#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include "catch.hpp"
