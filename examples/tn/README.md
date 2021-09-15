# Regular expression based text normalization for Chinese

For simplicity and ease of implementation, text normalization is basically done by rules and dictionaries. Here's an example.

## Run

```
. path.sh
bash run.sh
```

## Results

```
exp/
`-- normalized.txt

0 directories, 1 file
```

```
aff31f8aa08e2a7360228c9ce5886b98  exp/normalized.txt
```

```
今天的最低气温达到零下十度.
只要有四分之三十三的人同意，就可以通过决议。
一九四五年五月二日，苏联士兵在德国国会大厦上升起了胜利旗，象征着攻占柏林并战胜了纳粹德国。
四月十六日，清晨的战斗以炮击揭幕，数以千计的大炮和喀秋莎火箭炮开始炮轰德军阵地，炮击持续了数天之久。
如果剩下的百分之三十点六是过去，那么还有百分之六十九点四.
事情发生在二零二零年三月三十一日的上午八点.
警方正在找一支点二二口径的手枪。
欢迎致电中国联通，北京二零二二年冬奥会官方合作伙伴为您服务
充值缴费请按一，查询话费及余量请按二，跳过本次提醒请按井号键。
快速解除流量封顶请按星号键，腾讯王卡产品介绍、使用说明、特权及活动请按九，查询话费、套餐余量、积分及活动返款请按一，手机上网流量开通及取消请按二，查���本机号码及本号所使用套餐请按四，密码修改及重置请按五，紧急开机请按六，挂失请按七，查询充值记录请按八，其它自助服务及工服务请按零
```
