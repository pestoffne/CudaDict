#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>

#include "common.h"

#include "cpu.h"
#include "gpu.h"

using namespace std;

unsigned long file_size(const char *path)
{
	ifstream in(path, ifstream::ate | ifstream::binary);
	return in.tellg();
}

const char *read(const char *path, unsigned long size)
{
	unsigned long text_size = file_size(path);
	char *text = new char[text_size];

	ifstream ifs(path, ifstream::binary);

	if (!ifs) {
#if LANG_RU
		fprintf(stderr, "Невозможно прочитать файл.\n");
#else
		fprintf(stderr, "Can not read file.\n");
#endif
		exit(EXIT_FAILURE);
	}

	const unsigned long buffsize = 1024;

	ifs.read(text, text_size % buffsize);

	for (unsigned long i = text_size % buffsize; i < text_size; i += buffsize) {
		ifs.read(text + i, buffsize);
	}

	ifs.close();
	return text;
}

void print_dict(dict_t dict)
{
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

void print_diff(dict_t dict_cpu, dict_t dict_gpu)
{
	bool equal = true;

	for (int i = 0; i < 256; i++) {
		if (dict_cpu[i] != dict_gpu[i]) {
			if (equal) {
				printf("char      cpu      gpu\n");
			}

			printf(" '%1c' %8d %8d\n", i - 128, dict_cpu[i], dict_gpu[i]);
			equal = false;
		}
	}

#if LANG_RU
	printf("Cловари %s.\n", equal ? "совпадают" : "отличаются");
#else
	printf("Dicts are%s equal.\n", equal ? " " : "not");
#endif
}

int main(int argc, char **argv)
{
	UNUSED(argc);
	UNUSED(argv);

	const char *path = "/home/evgeny/Документы/Test/500";
	unsigned long full_text_size = file_size(path);
	const char *full_text = read(path, full_text_size);

	printf("%s (%.1f MiB)\n", path, static_cast<float>(full_text_size) / 1024 / 1024);

	time_t begin_t, end_t;

	begin_t = clock();
	dict_t dict_gpu = process_gpu(full_text, full_text_size);
	end_t = clock();
#if LANG_RU
	printf("Время выполнения алгоритма на GPU = %.3f секунд.\n",
#else
	printf("Process time on GPU = %.3f seconds.\n",
#endif
		   static_cast<float>(end_t - begin_t) / CLOCKS_PER_SEC);

	begin_t = clock();
	dict_t dict_cpu = process_cpu(full_text, full_text_size);
	end_t = clock();
#if LANG_RU
	printf("Время выполнения алгоритма на CPU = %.3f секунд.\n",
#else
	printf("Process time on CPU = %.3f seconds.\n",
#endif
		   static_cast<float>(end_t - begin_t) / CLOCKS_PER_SEC);

	delete full_text;

	print_diff(dict_cpu, dict_gpu);

#if PRINT_GPU_DICT
#if LANG_RU
	printf("Часть словаря GPU:\n");
#else
	printf("Part of GPU dict:\n");
#endif
	print_dict(dict_gpu);
#endif

	return 0;
}
