#include <Python.h>

#include <string.h>

char *PyMem_Strndup(const char *str, size_t len) {
    if (str != NULL) {
        char *copy = PyMem_New(char, len + 1);
        if (copy != NULL) memcpy(copy, str, len);
        copy[len] = '\0';
        return copy;
    }
    return NULL;
}

char *PyMem_Strdup(const char *str) { return PyMem_Strndup(str, strlen(str)); }

char *reprn(char *str, size_t len) {
    static char strings[10240];
    static size_t current = 0;
    size_t reqlen = 2;
    char c, *out, *write, *begin = str, *end = str + len;
    while (begin < end) {
        c = *begin;
        if (c == '\'') {
            reqlen += 2;
        } else if (c == '\r') {
            reqlen += 2;
        } else if (c == '\n') {
            reqlen += 2;
        } else if (c == '\t') {
            reqlen += 2;
        } else if (c < ' ') {
            reqlen += 3;
        } else {
            reqlen++;
        }
        begin++;
    }
    if (reqlen > 10240) {
        reqlen = 10240;
    }
    if (current + reqlen > 10240) {
        current = 0;
    }
    begin = str;
    end = str + len;
    out = write = strings + current;
    *write++ = '\'';
    while (begin < end) {
        c = *begin;
        if (c == '\'') {
            if (write + 5 >= strings + 10240) break;
            sprintf(write, "\\'");
            write += 2;
        } else if (c == '\r') {
            if (write + 5 >= strings + 10240) break;
            sprintf(write, "\\r");
            write += 2;
        } else if (c == '\n') {
            if (write + 5 >= strings + 10240) break;
            sprintf(write, "\\n");
            write += 2;
        } else if (c == '\t') {
            if (write + 5 >= strings + 10240) break;
            sprintf(write, "\\t");
            write += 2;
        } else if (c < ' ') {
            if (write + 6 >= strings + 10240) break;
            sprintf(write, "\\x%02x", c);
            write += 3;
        } else {
            if (write + 4 >= strings + 10240) break;
            *write++ = c;
        }
        begin++;
    }
    *write++ = '\'';
    *write++ = '\0';
    current += (size_t)(write - out);
    return out;
}

char *repr(char *str) { return reprn(str, strlen(str)); }
