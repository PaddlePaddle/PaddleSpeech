#ifndef _DICT_H_
#define _DICT_H_

#include "word.h"

/**
 * A dictionary is a hash table of
 *  - key: string
 *  - value: word
 *
 * Dictionary data can be loaded from files. Two type of dictionary
 * files are supported:
 *  - character file: Each line contains a number and a character,
 *                    the number is the frequency of the character.
 *                    The frequency should NOT exceeds 65535.
 *  - word file:      Each line contains a number and a word, the
 *                    number is the character count of the word.
 */

namespace rmmseg {
/* Instead of making a class with only one instance, i'll not
 * bother to make it a class here. */

namespace dict {
void add(Word *word);
bool load_chars(const char *filename);
bool load_words(const char *filename);
Word *get(const char *str, int len);
}
}

#endif /* _DICT_H_ */
