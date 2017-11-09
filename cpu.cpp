#include <fstream>

#include "cpu.h"

dict_t process_cpu(const char *path)
{
	ifstream ifs(path, ifstream::binary);

	if (!ifs) {
		fprintf(stderr, "Невозможно прочитать файл.\n");
		exit(EXIT_FAILURE);
	}

	dict_t dict = new unsigned int[256];

	for (int i = 0; i < 256; i++) {
		dict[i] = 0;
	}

	const int buffsize = 1024;  // 1 KiB
	char *buffer = new char[buffsize];

	for (int i = 0; i < buffsize; i++) {
		buffer[i] = '\0';
	}

	do {
		ifs.read(buffer, buffsize);

		for (int i = 0; i < buffsize; i++) {
			dict[buffer[i] + 128]++;
			buffer[i] = '\0';
		}
	} while (ifs);

	ifs.close();

	//dict['\0' + 128] = 0;

	return dict;
}
