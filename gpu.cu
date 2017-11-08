#include <fstream>

#include <cuda_runtime.h>

#include "common.h"

const int MAX_THREAD_PER_BLOCK = 1024;  // Для моей видеокарты (GeForce GT 730)
const int THREADS_PER_BLOCK = 2;  // Равно значению blockDim.x
const int BLOCK_PER_GRID = 2;
const int THREADS_PER_GRID = THREADS_PER_BLOCK * BLOCK_PER_GRID;
const int BUFFER_SIZE = 64; //1048576;  // 1 MiB
const int DICTS_SIZE = THREADS_PER_GRID * 256 * sizeof(unsigned int);

#define DIVIDE_CEIL(x, y) (((x) + (y) - 1) / (y))

__global__ void kernel(const char *buff, dict_t dicts)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	for (int j = i; j < BUFFER_SIZE; j += THREADS_PER_GRID) {
		dicts[buff[j] + 128 + 256 * i]++;
	}
}

__host__ dict_t process_gpu(const char *path)
{
	ifstream ifs(path, ifstream::binary);

	if (!ifs) {
		fprintf(stderr, "Невозможно прочитать файл.\n");
		exit(EXIT_FAILURE);
	}

	cudaError_t err = cudaSuccess;

	dict_t d_dicts = NULL;
	err = cudaMalloc((void **)&d_dicts, DICTS_SIZE);

	if (err) {
		fprintf(stderr,
				"Ошибка при выделении памяти для словаря на видеокарте: %s\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	dict_t h_dicts = new unsigned int[DICTS_SIZE];

	for (int i = 0; i < DICTS_SIZE; i++) {
		h_dicts[i] = 0;
	}

	err = cudaMemcpy(d_dicts, h_dicts, DICTS_SIZE, cudaMemcpyHostToDevice);

	if (err) {
		fprintf(stderr, "Ошибка при обнулении словаря на видеокарте: %s\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	delete h_dicts;

	char *d_buff = NULL;
	err = cudaMalloc((void **)&d_buff, BUFFER_SIZE);

	if (err) {
		fprintf(stderr,
				"Ошибка при выделении памяти для буфера на видеокарте: %s\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	char *h_buff = new char[BUFFER_SIZE];

	for (int i = 0; i < BUFFER_SIZE; i++) {
		h_buff[i] = '\0';
	}

	for (;;) {
		ifs.read(h_buff, BUFFER_SIZE);

		err = cudaMemcpy(d_buff, h_buff, BUFFER_SIZE, cudaMemcpyHostToDevice);

		if (err) {
			fprintf(stderr, "Ошибка при копировании буфера из оперативной памяти в"
					" память видеокарты: %s\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		kernel<<<BLOCK_PER_GRID, THREADS_PER_BLOCK>>>(d_buff, d_dicts);
		err = cudaGetLastError();

		if (err) {
			fprintf(stderr, "Ошибка при запуске вычислений на видеокарте: %s\n",
					cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		if (!ifs) {
			break;
		}

		for (int i = 0; i < BUFFER_SIZE; i++) {
			h_buff[i] = '\0';
		}
	}

	ifs.close();
	delete h_buff;

	err = cudaFree(d_buff);

	if (err) {
		fprintf(stderr, "Ошибка при освобождении буфера: %s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	h_dicts = new unsigned int[DICTS_SIZE];
	err = cudaMemcpy(h_dicts, d_dicts, DICTS_SIZE, cudaMemcpyDeviceToHost);

	if (err) {
		fprintf(stderr,
				"Ошибка при копировании словаря из памяти видеокарты в оперативную память: %s\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_dicts);

	if (err) {
		fprintf(stderr, "Ошибка при освобождении словаря: %s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	dict_t dict = new unsigned int[256];

	for (int i = 0; i < 256; i++) {
		dict[i] = h_dicts[i];

		for (int j = 1; j < THREADS_PER_GRID; j++) {
			dict[i] += h_dicts[i + 256 * j];
		}
	}

	delete h_dicts;

	return dict;
}

__host__ int main(int argc, char **argv)
{
#if 0
	printf("THREADS_PER_BLOCK = %d\n", THREADS_PER_BLOCK);
	printf("BLOCK_PER_GRID = %d\n", BLOCK_PER_GRID);
	printf("THREADS_PER_GRID = %d\n", THREADS_PER_GRID);
	printf("BUFFER_SIZE = %d\n", BUFFER_SIZE);
	printf("DICTS_SIZE = %d\n", DICTS_SIZE);
	printf("sizeof(unsigned int) = %lu\n", sizeof(unsigned int));
#endif
	run(process_gpu, argc, argv);
	return 0;
}
