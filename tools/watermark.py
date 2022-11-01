# add watermark for text
def watermark(content, pattern):
    m = list(zip(pattern * (len(content) // len(pattern) + 1), content))
    return ''.join([x for t in m
                    for x in t] + [pattern[len(content) % len(pattern)]])


# remove cyclic watermark in text
def iwatermark(content):
    e = [x for i, x in enumerate(content) if i % 2 == 0]
    o = [x for i, x in enumerate(content) if i % 2 != 0]
    for i in range(1, len(e) // 2 + 1):
        if e[i:] == e[:-i]:
            return ''.join(o)
    return ''.join(e)


if __name__ == "__main__":
    print(watermark('跟世龙对齐 Triton 开发计划', 'hbzs'))
    print(iwatermark('h跟b世z龙s对h齐b zTsrhibtzosnh b开z发s计h划b'))
