// Copyright (c) code is from
// https://blog.csdn.net/huixingshao/article/details/45969887.

#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
using namespace std;

#pragma once

#ifdef _MSC_VER
#pragma region ParseIniFile
#endif

/*
 * \brief Generic configuration Class
 *
 */
class Config {
    // Data
  protected:
    std::string m_Delimiter;  //!< separator between key and value
    std::string m_Comment;    //!< separator between value and comments
    std::map<std::string, std::string>
        m_Contents;  //!< extracted keys and values

    typedef std::map<std::string, std::string>::iterator mapi;
    typedef std::map<std::string, std::string>::const_iterator mapci;
    // Methods
  public:
    Config(std::string filename,
           std::string delimiter = "=",
           std::string comment = "#");
    Config();
    template <class T>
    T Read(const std::string& in_key) const;  //!< Search for key and read value
    //! or optional default value, call
    //! as read<T>
    template <class T>
    T Read(const std::string& in_key, const T& in_value) const;
    template <class T>
    bool ReadInto(T* out_var, const std::string& in_key) const;
    template <class T>
    bool ReadInto(T* out_var,
                  const std::string& in_key,
                  const T& in_value) const;
    bool FileExist(std::string filename);
    void ReadFile(std::string filename,
                  std::string delimiter = "=",
                  std::string comment = "#");

    // Check whether key exists in configuration
    bool KeyExists(const std::string& in_key) const;

    // Modify keys and values
    template <class T>
    void Add(const std::string& in_key, const T& in_value);
    void Remove(const std::string& in_key);

    // Check or change configuration syntax
    std::string GetDelimiter() const { return m_Delimiter; }
    std::string GetComment() const { return m_Comment; }
    std::string SetDelimiter(const std::string& in_s) {
        std::string old = m_Delimiter;
        m_Delimiter = in_s;
        return old;
    }
    std::string SetComment(const std::string& in_s) {
        std::string old = m_Comment;
        m_Comment = in_s;
        return old;
    }

    // Write or read configuration
    friend std::ostream& operator<<(std::ostream& os, const Config& cf);
    friend std::istream& operator>>(std::istream& is, Config& cf);

  protected:
    template <class T>
    static std::string T_as_string(const T& t);
    template <class T>
    static T string_as_T(const std::string& s);
    static void Trim(std::string* inout_s);


    // Exception types
  public:
    struct File_not_found {
        std::string filename;
        explicit File_not_found(const std::string& filename_ = std::string())
            : filename(filename_) {}
    };
    struct Key_not_found {  // thrown only by T read(key) variant of read()
        std::string key;
        explicit Key_not_found(const std::string& key_ = std::string())
            : key(key_) {}
    };
};

/* static */
template <class T>
std::string Config::T_as_string(const T& t) {
    // Convert from a T to a string
    // Type T must support << operator
    std::ostringstream ost;
    ost << t;
    return ost.str();
}


/* static */
template <class T>
T Config::string_as_T(const std::string& s) {
    // Convert from a string to a T
    // Type T must support >> operator
    T t;
    std::istringstream ist(s);
    ist >> t;
    return t;
}


/* static */
template <>
inline std::string Config::string_as_T<std::string>(const std::string& s) {
    // Convert from a string to a string
    // In other words, do nothing
    return s;
}


/* static */
template <>
inline bool Config::string_as_T<bool>(const std::string& s) {
    // Convert from a string to a bool
    // Interpret "false", "F", "no", "n", "0" as false
    // Interpret "true", "T", "yes", "y", "1", "-1", or anything else as true
    bool b = true;
    std::string sup = s;
    for (std::string::iterator p = sup.begin(); p != sup.end(); ++p)
        *p = toupper(*p);  // make string all caps
    if (sup == std::string("FALSE") || sup == std::string("F") ||
        sup == std::string("NO") || sup == std::string("N") ||
        sup == std::string("0") || sup == std::string("NONE"))
        b = false;
    return b;
}


template <class T>
T Config::Read(const std::string& key) const {
    // Read the value corresponding to key
    mapci p = m_Contents.find(key);
    if (p == m_Contents.end()) throw Key_not_found(key);
    return string_as_T<T>(p->second);
}


template <class T>
T Config::Read(const std::string& key, const T& value) const {
    // Return the value corresponding to key or given default value
    // if key is not found
    mapci p = m_Contents.find(key);
    if (p == m_Contents.end()) {
        printf("%s = %s(default)\n", key.c_str(), T_as_string(value).c_str());
        return value;
    } else {
        printf("%s = %s\n", key.c_str(), T_as_string(p->second).c_str());
        return string_as_T<T>(p->second);
    }
}


template <class T>
bool Config::ReadInto(T* var, const std::string& key) const {
    // Get the value corresponding to key and store in var
    // Return true if key is found
    // Otherwise leave var untouched
    mapci p = m_Contents.find(key);
    bool found = (p != m_Contents.end());
    if (found) *var = string_as_T<T>(p->second);
    return found;
}


template <class T>
bool Config::ReadInto(T* var, const std::string& key, const T& value) const {
    // Get the value corresponding to key and store in var
    // Return true if key is found
    // Otherwise set var to given default
    mapci p = m_Contents.find(key);
    bool found = (p != m_Contents.end());
    if (found)
        *var = string_as_T<T>(p->second);
    else
        var = value;
    return found;
}


template <class T>
void Config::Add(const std::string& in_key, const T& value) {
    // Add a key with given value
    std::string v = T_as_string(value);
    std::string key = in_key;
    Trim(&key);
    Trim(&v);
    m_Contents[key] = v;
    return;
}

Config::Config(string filename, string delimiter, string comment)
    : m_Delimiter(delimiter), m_Comment(comment) {
    // Construct a Config, getting keys and values from given file

    std::ifstream in(filename.c_str());

    if (!in) throw File_not_found(filename);

    in >> (*this);
}


Config::Config() : m_Delimiter(string(1, '=')), m_Comment(string(1, '#')) {
    // Construct a Config without a file; empty
}


bool Config::KeyExists(const string& key) const {
    // Indicate whether key is found
    mapci p = m_Contents.find(key);
    return (p != m_Contents.end());
}


/* static */
void Config::Trim(string* inout_s) {
    // Remove leading and trailing whitespace
    static const char whitespace[] = " \n\t\v\r\f";
    inout_s->erase(0, inout_s->find_first_not_of(whitespace));
    inout_s->erase(inout_s->find_last_not_of(whitespace) + 1U);
}


std::ostream& operator<<(std::ostream& os, const Config& cf) {
    // Save a Config to os
    for (Config::mapci p = cf.m_Contents.begin(); p != cf.m_Contents.end();
         ++p) {
        os << p->first << " " << cf.m_Delimiter << " ";
        os << p->second << std::endl;
    }
    return os;
}

void Config::Remove(const string& key) {
    // Remove key and its value
    m_Contents.erase(m_Contents.find(key));
    return;
}

std::istream& operator>>(std::istream& is, Config& cf) {
    // Load a Config from is
    // Read in keys and values, keeping internal whitespace
    typedef string::size_type pos;
    const string& delim = cf.m_Delimiter;  // separator
    const string& comm = cf.m_Comment;     // comment
    const pos skip = delim.length();       // length of separator

    string nextline = "";  // might need to read ahead to see where value ends

    while (is || nextline.length() > 0) {
        // Read an entire line at a time
        string line;
        if (nextline.length() > 0) {
            line = nextline;  // we read ahead; use it now
            nextline = "";
        } else {
            std::getline(is, line);
        }

        // Ignore comments
        line = line.substr(0, line.find(comm));

        // Parse the line if it contains a delimiter
        pos delimPos = line.find(delim);
        if (delimPos < string::npos) {
            // Extract the key
            string key = line.substr(0, delimPos);
            line.replace(0, delimPos + skip, "");

            // See if value continues on the next line
            // Stop at blank line, next line with a key, end of stream,
            // or end of file sentry
            bool terminate = false;
            while (!terminate && is) {
                std::getline(is, nextline);
                terminate = true;

                string nlcopy = nextline;
                Config::Trim(&nlcopy);
                if (nlcopy == "") continue;

                nextline = nextline.substr(0, nextline.find(comm));
                if (nextline.find(delim) != string::npos) continue;

                nlcopy = nextline;
                Config::Trim(&nlcopy);
                if (nlcopy != "") line += "\n";
                line += nextline;
                terminate = false;
            }

            // Store key and value
            Config::Trim(&key);
            Config::Trim(&line);
            cf.m_Contents[key] = line;  // overwrites if key is repeated
        }
    }

    return is;
}
bool Config::FileExist(std::string filename) {
    bool exist = false;
    std::ifstream in(filename.c_str());
    if (in) exist = true;
    return exist;
}

void Config::ReadFile(string filename, string delimiter, string comment) {
    m_Delimiter = delimiter;
    m_Comment = comment;
    std::ifstream in(filename.c_str());

    if (!in) throw File_not_found(filename);

    in >> (*this);
}

#ifdef _MSC_VER
#pragma endregion ParseIniFIle
#endif
