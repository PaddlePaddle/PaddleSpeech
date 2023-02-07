// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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


#include "utils/strings.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>


TEST(StringTest, StrSplitTest) {
    using ::testing::ElementsAre;

    std::string test_str = "hello world";
    std::vector<std::string> outs = ppspeech::StrSplit(test_str, " \t");
    EXPECT_THAT(outs, ElementsAre("hello", "world"));
}


TEST(StringTest, StrJoinTest) {
    std::vector<std::string> ins{"hello", "world"};
    std::string out = ppspeech::StrJoin(ins, " ");
    EXPECT_THAT(out, "hello world");
}