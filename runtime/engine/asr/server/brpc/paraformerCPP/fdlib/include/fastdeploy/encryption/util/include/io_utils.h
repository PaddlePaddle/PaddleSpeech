// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#include <iostream>
#include <memory>
#include <vector>
#include <string>

#ifndef PADDLE_MODEL_PROTECT_IO_UTILS_H
#define PADDLE_MODEL_PROTECT_IO_UTILS_H
namespace fastdeploy {
namespace ioutil {

int read_file(const char* file_path, unsigned char** dataptr, size_t* sizeptr);

int read_with_pos_and_length(const char* file_path, unsigned char* dataptr,
                             size_t pos, size_t length);

int read_with_pos(const char* file_path, size_t pos, unsigned char** dataptr,
                  size_t* sizeptr);

int write_file(const char* file_path, const unsigned char* dataptr,
               size_t sizeptr);

int append_file(const char* file_path, const unsigned char* data, size_t len);

size_t read_file_size(const char* file_path);

int read_file_to_file(const char* src_path, const char* dst_path);

int dir_exist_or_mkdir(const char* dir);

/**
 * @return files.size()
 */
int read_dir_files(const char* dir_path,
                   std::vector<std::string>& files);  // NOLINT

}  // namespace ioutil
}  // namespace fastdeploy
#endif  // PADDLE_MODEL_PROTECT_IO_UTILS_H
