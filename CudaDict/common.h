#pragma once

#define UNUSED(x) (void)(x)

#define PRINT_GPU_DICT 1
#define MSVS 1

#if MSVS
#define LANG_RU 0

#define EXIT \
	getchar(); \
	exit(EXIT_FAILURE)

#define END \
	getchar(); \
	exit(EXIT_SUCCESS)
#else
#define LANG_RU 1
#define EXIT exit(EXIT_FAILURE)
#define END exit(EXIT_SUCCESS)
#endif
