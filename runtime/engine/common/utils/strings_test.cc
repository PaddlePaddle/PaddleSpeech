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

TEST(StringText, DelBlankTest) {
    std::string test_str = "我 今天     去 了 超市     花了   120 元。";
    std::string out_str = ppspeech::DelBlank(test_str);
    int ret = out_str.compare("我今天去了超市花了120元。");
    EXPECT_EQ(ret, 0);

    test_str = "how are you today";
    out_str = ppspeech::DelBlank(test_str);
    ret = out_str.compare("how are you today");
    EXPECT_EQ(ret, 0);

    test_str = "我 的 paper 在 哪里？";
    out_str = ppspeech::DelBlank(test_str);
    ret = out_str.compare("我的paper在哪里？");
    EXPECT_EQ(ret, 0);
}

TEST(StringTest, AddBlankTest) {
    std::string test_str = "how are you";
    std::string out_str = ppspeech::AddBlank(test_str);
    int ret = out_str.compare(" how  are  you ");
    EXPECT_EQ(ret, 0);

    test_str = "欢迎来到China。";
    out_str = ppspeech::AddBlank(test_str);
    ret = out_str.compare("欢迎来到 China 。");
    EXPECT_EQ(ret, 0);
}

TEST(StringTest, ReverseFractionTest) {
    std::string test_str = "<tag>3/1<tag>";
    std::string out_str = ppspeech::ReverseFraction(test_str);
    int ret = out_str.compare("1/3");
    std::cout<<out_str<<std::endl;
    EXPECT_EQ(ret, 0);

    test_str = "<tag>3/1<tag> <tag>100/10000<tag>";
    out_str = ppspeech::ReverseFraction(test_str);
    ret = out_str.compare("1/3 10000/100");
    std::cout<<out_str<<std::endl;
    EXPECT_EQ(ret, 0);
}
