#ifndef BASE_TYPE_CONVC_H
#define BASE_TYPE_CONVC_H

#include <string>
#include <locale>
#include <codecvt>


namespace speechnn {
// wstring to string
std::string wstring2utf8string(const std::wstring& str);
 
// string to wstring 
std::wstring utf8string2wstring(const std::string& str);

}

#endif  // BASE_TYPE_CONVC_H