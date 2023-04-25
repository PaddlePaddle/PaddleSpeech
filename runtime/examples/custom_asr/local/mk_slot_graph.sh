#!/bin/bash

# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

graph_slot=$1
dir=$2

[ -f path.sh ] && . ./path.sh

sym=$dir/../lang/words.txt
cat > $dir/address_slot.txt <<EOF
0 1 南山 南山
0 1 南京 南京
0 1 光明 光明   
0 1 龙岗 龙岗
0 1 北苑 北苑
0 1 北京 北京
0 1 酒店 酒店
0 1 合肥 合肥
0 1 望京搜后 望京搜后
0 1 地铁站 地铁站
0 1 海淀黄庄 海淀黄庄
0 1 佛山 佛山
0 1 广州 广州
0 1 苏州 苏州
0 1 百度大厦 百度大厦
0 1 龙泽苑东区 龙泽苑东区
0 1 首都机场 首都机场
0 1 朝来家园 朝来家园
0 1 深大 深大
0 1 双龙 双龙
0 1 公司 公司
0 1 上海 上海
0 1 家 家
0 1 机场 机场
0 1 华祝 华祝
0 1 上海虹桥 上海虹桥
0 2 检验 检验
2 1 中心 中心
0 3 苏州 苏州
3 1 街 街
3 8 高铁 高铁
8 1 站 站
0 4 杭州 杭州
4 1 东站 东站
4 1 <eps> <eps>
0 5 上海 上海
0 5 北京 北京
0 5 合肥 合肥
5 1 南站 南站
0 6 立水 立水
6 1 桥 桥
0 7 青岛 青岛
7 1 站 站
1
EOF

fstcompile --isymbols=$sym --osymbols=$sym $dir/address_slot.txt $dir/address_slot.fst
fstcompile --isymbols=$sym --osymbols=$sym $graph_slot/time_slot.txt $dir/time_slot.fst
fstcompile --isymbols=$sym --osymbols=$sym $graph_slot/date_slot.txt $dir/date_slot.fst
fstcompile --isymbols=$sym --osymbols=$sym $graph_slot/money_slot.txt $dir/money_slot.fst
fstcompile --isymbols=$sym --osymbols=$sym $graph_slot/year_slot.txt $dir/year_slot.fst
