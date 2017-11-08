#include <cstdio>
#include <cstdlib>

#include "common.h"

const char *parse(int argc, char **argv)
{
	if (argc != 2) {
		fprintf(stderr, "Требуется один аргумент.\n");
		exit(EXIT_FAILURE);
	}

	return argv[1];
}

void print_result(const char *path, dict_t dict)
{
	printf("%s:\n", path);

	unsigned int a;

	if (a = dict['\t' + 128]) {
		printf("\\t %d\n", a);
	}

	if (a = dict['\n' + 128]) {
		printf("\\n %d\n", a);
	}

	if (a = dict['\r' + 128]) {
		printf("\\r %d\n", a);
	}

	for (char c = 32; c < 127; c++) {
		if (a = dict[c + 128]) {
			printf("%c  %d\n", c, a);
		}
	}
}

void run(func_t process, int argc, char **argv)
{
	const char *path = parse(argc, argv);

	dict_t dict = process(path);

	print_result(path, dict);

	delete dict;
}
