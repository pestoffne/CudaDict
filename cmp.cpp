#include <cstdio>
#include <cstdlib>
#include <ctime>

#include "cpu.h"
#include "gpu.h"

const char *parse(int argc, char **argv)
{
	if (argc != 2) {
		fprintf(stderr, "Требуется один аргумент.\n");
		exit(EXIT_FAILURE);
	}

	return argv[1];
}

void print_diff(dict_t dict_cpu, dict_t dict_gpu)
{
	bool equal = true;

	printf("char      cpu      gpu\n");
	for (int i = 0; i < 256; i++) {
		if (dict_cpu[i] != dict_gpu[i]) {
			printf(" '%1c' %8d %8d\n", i - 128, dict_cpu[i], dict_gpu[i]);
			equal = false;
		}
	}

	printf("Cловари %s.\n", equal ? "совпадают" : "отличаются");
}

bool equal(dict_t a, dict_t b)
{
	bool c = true;

	for (int i = 0; i < 256; i++) {
		if (a[i] != b[i]) {
			fprintf(stderr, "dict[%d '%c']: cpu = %d, gpu = %d\n", i - 128, i - 128, a[i], b[i]);
			c = false;
		}
	}

	return c;
}

dict_t process_both(const char *path)
{
	time_t begin_t, end_t;

	begin_t = clock();
	dict_t dict_gpu = process_gpu(path);
	end_t = clock();
	printf("Время выполнения алгоритма на GPU = %.3f\n", static_cast<float>(end_t - begin_t) / CLOCKS_PER_SEC);

	dict_t dict_cpu = process_cpu(path);

	print_diff(dict_cpu, dict_gpu);
}

int main(int argc, char **argv)
{
	const char *path = parse(argc, argv);
	printf("%s:\n", path);

	process_both(path);

	return 0;
}
