# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
# limitations under the License.
import re
import tempfile

from paddlespeech.t2s.frontend.mix_frontend import MixFrontend

# mix zh & en phonemes
phone_id_str = """
<pad> 0
<unk> 1
AA0 2
AA1 3
AA2 4
AE0 5
AE1 6
AE2 7
AH0 8
AH1 9
AH2 10
AO0 11
AO1 12
AO2 13
AW0 14
AW1 15
AW2 16
AY0 17
AY1 18
AY2 19
B 20
CH 21
D 22
DH 23
EH0 24
EH1 25
EH2 26
ER0 27
ER1 28
ER2 29
EY0 30
EY1 31
EY2 32
F 33
G 34
HH 35
IH0 36
IH1 37
IH2 38
IY0 39
IY1 40
IY2 41
JH 42
K 43
L 44
M 45
N 46
NG 47
OW0 48
OW1 49
OW2 50
OY0 51
OY1 52
OY2 53
P 54
R 55
S 56
SH 57
T 58
TH 59
UH0 60
UH1 61
UH2 62
UW0 63
UW1 64
UW2 65
V 66
W 67
Y 68
Z 69
ZH 70
a1 71
a2 72
a3 73
a4 74
a5 75
ai1 76
ai2 77
ai3 78
ai4 79
ai5 80
air2 81
air3 82
air4 83
an1 84
an2 85
an3 86
an4 87
an5 88
ang1 89
ang2 90
ang3 91
ang4 92
ang5 93
angr2 94
angr4 95
anr1 96
anr3 97
anr4 98
ao1 99
ao2 100
ao3 101
ao4 102
ao5 103
aor1 104
aor3 105
aor4 106
aor5 107
ar2 108
ar3 109
ar4 110
ar5 111
b 112
c 113
ch 114
d 115
e1 116
e2 117
e3 118
e4 119
e5 120
ei1 121
ei2 122
ei3 123
ei4 124
ei5 125
eir4 126
en1 127
en2 128
en3 129
en4 130
en5 131
eng1 132
eng2 133
eng3 134
eng4 135
eng5 136
engr4 137
enr1 138
enr2 139
enr3 140
enr4 141
enr5 142
er1 143
er2 144
er3 145
er4 146
er5 147
f 148
g 149
h 150
i1 151
i2 152
i3 153
i4 154
i5 155
ia1 156
ia2 157
ia3 158
ia4 159
ia5 160
ian1 161
ian2 162
ian3 163
ian4 164
ian5 165
iang1 166
iang2 167
iang3 168
iang4 169
iang5 170
iangr4 171
ianr1 172
ianr2 173
ianr3 174
ianr4 175
ianr5 176
iao1 177
iao2 178
iao3 179
iao4 180
iao5 181
iaor1 182
iaor2 183
iaor3 184
iaor4 185
iar1 186
iar3 187
iar4 188
ie1 189
ie2 190
ie3 191
ie4 192
ie5 193
ii1 194
ii2 195
ii3 196
ii4 197
ii5 198
iii1 199
iii2 200
iii3 201
iii4 202
iii5 203
iiir1 204
iiir4 205
iir2 206
in1 207
in2 208
in3 209
in4 210
in5 211
ing1 212
ing2 213
ing3 214
ing4 215
ing5 216
ingr1 217
ingr2 218
ingr3 219
ingr4 220
inr1 221
inr4 222
io1 223
io3 224
io5 225
iong1 226
iong2 227
iong3 228
iong4 229
iong5 230
iou1 231
iou2 232
iou3 233
iou4 234
iou5 235
iour1 236
iour2 237
iour3 238
iour4 239
ir1 240
ir2 241
ir3 242
ir4 243
ir5 244
j 245
k 246
l 247
m 248
n 249
o1 250
o2 251
o3 252
o4 253
o5 254
ong1 255
ong2 256
ong3 257
ong4 258
ong5 259
ongr4 260
or2 261
ou1 262
ou2 263
ou3 264
ou4 265
ou5 266
our2 267
our3 268
our4 269
our5 270
p 271
q 272
r 273
s 274
sh 275
sil 276
sp 277
spl 278
spn 279
t 280
u1 281
u2 282
u3 283
u4 284
u5 285
ua1 286
ua2 287
ua3 288
ua4 289
ua5 290
uai1 291
uai2 292
uai3 293
uai4 294
uai5 295
uair4 296
uan1 297
uan2 298
uan3 299
uan4 300
uan5 301
uang1 302
uang2 303
uang3 304
uang4 305
uang5 306
uangr4 307
uanr1 308
uanr2 309
uanr3 310
uanr4 311
uanr5 312
uar1 313
uar2 314
uar4 315
uei1 316
uei2 317
uei3 318
uei4 319
uei5 320
ueir1 321
ueir2 322
ueir3 323
ueir4 324
uen1 325
uen2 326
uen3 327
uen4 328
uen5 329
ueng1 330
ueng2 331
ueng3 332
ueng4 333
uenr1 334
uenr2 335
uenr3 336
uenr4 337
uo1 338
uo2 339
uo3 340
uo4 341
uo5 342
uor1 343
uor2 344
uor3 345
uor5 346
ur1 347
ur2 348
ur3 349
ur4 350
ur5 351
v1 352
v2 353
v3 354
v4 355
v5 356
van1 357
van2 358
van3 359
van4 360
van5 361
vanr1 362
vanr2 363
vanr3 364
vanr4 365
ve1 366
ve2 367
ve3 368
ve4 369
ve5 370
ver3 371
ver4 372
vn1 373
vn2 374
vn3 375
vn4 376
vn5 377
vnr2 378
vr3 379
x 380
z 381
zh 382
, 383
. 384
? 385
! 386
<eos> 387
"""

if __name__ == '__main__':
    with tempfile.NamedTemporaryFile(mode='wt') as f:
        phone_ids = phone_id_str.split()
        for phone, id in zip(phone_ids[::2], phone_ids[1::2]):
            f.write(f"{phone} {id}")
            f.write('\n')
            f.flush()

        frontend = MixFrontend(phone_vocab_path=f.name)

        text = "hello, 我爱北京天安们，what about you."
        print(text)
        # [('hello, ', 'en'), ('我爱北京天安们，', 'zh'), ('what about you.', 'en')]
        segs = frontend.split_by_lang(text)
        print(segs)

        text = "hello?!!我爱北京天安们，what about you."
        print(text)
        # [('hello?!!', 'en'), ('我爱北京天安们，', 'zh'), ('what about you.', 'en')]
        segs = frontend.split_by_lang(text)
        print(segs)

        text = "<speak> hello，我爱北京天安们，what about you."
        print(text)
        # [('<speak> hello，', 'en'), ('我爱北京天安们，', 'zh'), ('what about you.', 'en')]
        segs = frontend.split_by_lang(text)
        print(segs)

        # 对于SSML的xml标记处理不好。需要先解析SSML，后处理中英的划分。
        text = "<speak>我们的声学模型使用了 Fast Speech Two。前浪<say-as pinyin='dao3'>倒</say-as>在沙滩上,沙滩上倒了一堆<say-as pinyin='tu3'>土</say-as>。 想象<say-as pinyin='gan1 gan1'>干干</say-as>的树干<say-as pinyin='dao3'>倒</say-as>了, 里面有个干尸，不知是被谁<say-as pinyin='gan4'>干</say-as>死的。</speak>"
        print(text)
        # [('<speak>', 'en'), ('我们的声学模型使用了 ', 'zh'), ('Fast Speech Two。', 'en'), ('前浪<', 'zh'), ("say-as pinyin='dao3'>", 'en'), ('倒</', 'zh'), ('say-as>', 'en'), ('在沙滩上,沙滩上倒了一堆<', 'zh'), ("say-as pinyin='tu3'>", 'en'), ('土</', 'zh'), ('say-as>。 ', 'en'), ('想象<', 'zh'), ("say-as pinyin='gan1 gan1'>", 'en'), ('干干</', 'zh'), ('say-as>', 'en'), ('的树干<', 'zh'), ("say-as pinyin='dao3'>", 'en'), ('倒</', 'zh'), ('say-as>', 'en'), ('了, 里面有个干尸，不知是被谁<', 'zh'), ("say-as pinyin='gan4'>", 'en'), ('干</', 'zh'), ('say-as>', 'en'), ('死的。</', 'zh'), ('speak>', 'en')]
        segs = frontend.split_by_lang(text)
        print(segs)
