#include <algorithm>
#include <fstream>

#include "cpu.h"

using namespace std;

dict_t process_cpu(const char *text, unsigned long text_size)
{
	dict_t dict = new unsigned int[256];
	fill(dict, dict + 256, 0);

	for (unsigned long i = 0; i < text_size; i++) {
		dict[text[i] + 128]++;
	}

	return dict;
}
