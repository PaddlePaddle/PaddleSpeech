

#include <pybind11/pybind11.h>

#include <iostream>
#include <string>

#include "algor.h"
#include "dict.h"
#include "token.h"
#include "utils.h"

namespace py = pybind11;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)


struct Token {
    const char *text;
    int offset;
    int length;
};


PYBIND11_MODULE(mmseg, m) {
    // String literal: https://en.cppreference.com/w/cpp/language/string_literal
    m.doc() = R"pbdoc(
    MMSeg pybind
    )pbdoc";

    m.def("load_chars", [](const char *path) {
        if (rmmseg::dict::load_chars(path)) {
            return true;
        }
        return false;
    });

    m.def("load_words", [](const char *path) {
        if (rmmseg::dict::load_words(path)) {
            return true;
        }
        return false;
    });

    m.def("add", [](const char *word, int len, int freq) {
        /*
        * Add a word to the in-memory dictionary.
        *
        * - word is a String.
        * - length is number of characters (not number of
        * bytes) of the
        *   word to be added.
        * - freq is the frequency of the word. This is only
        * used when
        *   it is a one-character word.
        */
        rmmseg::Word *w = rmmseg::make_word(word, len, freq, strlen(word));
        rmmseg::dict::add(w);
    });

    m.def("has_word", [](const char *word) {
        if (rmmseg::dict::get(word, static_cast<int>(strlen(word)))) {
            return true;
        }
        return false;
    });

    py::class_<rmmseg::Token>(m, "Token")
        .def(py::init([](std::string str) {
            return rmmseg::Token(str.c_str(), str.size());
        }))
        .def_property_readonly("text",
                               [](rmmseg::Token &self) {
                                   return std::string(self.text, self.length);
                               })
        .def_readonly("length", &rmmseg::Token::length)
        .def("__repr__",
             [](rmmseg::Token &self) {
                 return "<Token " + std::string(self.text, self.length) + " " +
                        std::to_string(self.length) + ">";
             })
        .def("__str__", [](rmmseg::Token &self) {
            return std::string(self.text, self.length);
        });


    py::class_<rmmseg::Algorithm>(m, "Algorithm")
        //.def(py::init<const char *, int>(),  py::keep_alive<1, 2>())
        .def(py::init([](std::string str) { return rmmseg::Algorithm(str); }))
        .def("get_text",
             [](rmmseg::Algorithm &self) { return self.get_text(); },
             py::return_value_policy::reference)
        .def("next_token",
             [](rmmseg::Algorithm &self) { return self.next_token(); });

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}