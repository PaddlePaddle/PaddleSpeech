import mmseg
import os

print(mmseg.load_chars('mmseg/data/chars.dic'))
print(mmseg.load_words('mmseg/data/words.dic'))
print(mmseg.has_word('我'))

print(dir(mmseg.Token))
print(dir(mmseg.Algorithm))


string="我是中国人武汉长江大桥".encode('utf8')
string="我是中国人武汉长江大桥"

t = mmseg.Token(string)
print(t)
print(t.text)

print("="*20)


a = mmseg.Algorithm(string)
print(a.get_text())
print(a.get_text().encode('utf8'))

print("="*20)
print(string)
while True:
    tk = a.next_token()
    if tk.length == 0:
        break
    c = string
    #print(a.get_text())
    #print(tk.length)
    print(tk.text)
