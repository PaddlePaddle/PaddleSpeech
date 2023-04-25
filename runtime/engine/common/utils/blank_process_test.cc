#include "utils/blank_process.h"

#include <gtest/gtest.h>
#include <gmock/gmock.h>

TEST(BlankProcess, BlankProcessTest) {
    std::string test_str = "我 今天 去 了 超市 花了 120 元。";
    std::string out_str = ppspeech::BlankProcess(test_str);
    int ret = out_str.compare("我今天去了超市花了120元。");
    EXPECT_EQ(ret, 0);

    test_str = "how are you today";
    out_str = ppspeech::BlankProcess(test_str);
    ret = out_str.compare("how are you today");
    EXPECT_EQ(ret, 0);

    test_str = "我 的 paper 在 哪里？";
    out_str = ppspeech::BlankProcess(test_str);
    ret = out_str.compare("我的paper在哪里？");
    EXPECT_EQ(ret, 0);

    test_str = "我 今天     去 了 超市     花了   120 元。";
    out_str = ppspeech::BlankProcess(test_str);
    ret = out_str.compare("我今天去了超市花了120元。");
    EXPECT_EQ(ret, 0);
}