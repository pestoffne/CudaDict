#include <cstdio>
#include <cstdlib>
#include <ctime>

#include "cpu.h"

const char *parse(int argc, char **argv)
{
	if (argc != 2) {
		fprintf(stderr, "Требуется один аргумент.\n");
		exit(EXIT_FAILURE);
	}

	return argv[1];
}

int main(int argc, char **argv)
{
	const char *path = parse(argc, argv);
	printf("%s:\n", path);

	time_t begin_t, end_t;

	begin_t = clock();
	dict_t dict = process_cpu(path);
	end_t = clock();
	printf("Время выполнения алгоритма на CPU = %.3f секунд.\n",
		   static_cast<float>(end_t - begin_t) / CLOCKS_PER_SEC);

	return 0;
}
