#pragma once

using namespace std;

typedef unsigned int *dict_t;
typedef dict_t func_t(const char *);

const char *parse(int argc, char **argv);
void print_result(const char *path, dict_t dict);
void run(func_t process, int argc, char **argv);
