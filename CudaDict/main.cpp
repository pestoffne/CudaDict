#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>

#include "common.h"

#include "cpu.h"
#include "gpu.h"

using namespace std;

#define TIME_INTERVAL(b, e) (static_cast<float>((e) - (b)) / CLOCKS_PER_SEC)

unsigned long file_size(const char *path)
{
	ifstream in(path, ifstream::ate | ifstream::binary);

	if (!in) {
#if LANG_RU
		fprintf(stderr, "Невозможно прочитать файл.\n");
#else
		fprintf(stderr, "Can not read file.\n");
#endif
		EXIT;
	}

	return in.tellg();
}

const char *read(const char *path, unsigned long file_size)
{
	char *text = new char[file_size];

	ifstream ifs(path, ifstream::binary);

	const unsigned long buff_size = 1024;

	ifs.read(text, file_size % buff_size);

	for (unsigned long i = file_size % buff_size; i < file_size; i += buff_size) {
		ifs.read(text + i, buff_size);
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
	printf("Dicts are%s equal.\n", equal ? "" : " not");
#endif
}

int main(int argc, char **argv)
{
	UNUSED(argc);
	UNUSED(argv);

	const char *path = "main.cpp";
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
		   TIME_INTERVAL(begin_t, end_t));

	begin_t = clock();
	dict_t dict_cpu = process_cpu(full_text, full_text_size);
	end_t = clock();
#if LANG_RU
	printf("Время выполнения алгоритма на CPU = %.3f секунд.\n",
#else
	printf("Process time on CPU = %.3f seconds.\n",
#endif
		   TIME_INTERVAL(begin_t, end_t));

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

	END;
}
