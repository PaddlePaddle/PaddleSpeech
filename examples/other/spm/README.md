# [SentencePiece Model](https://github.com/google/sentencepiece)

## Run
Train a `spm` model for English tokenizer.

```
. path.sh
bash run.sh
```

## Results

```
data/
└── lang_char
    ├── input.bpe
    ├── input.decode
    ├── input.txt
    ├── train_unigram100.model
    ├── train_unigram100_units.txt
    └── train_unigram100.vocab

1 directory, 6 files
```

```
b5a230c26c61db5c36f34e503102f936  data/lang_char/input.bpe
ec5a9b24acc35469229e41256ceaf77d  data/lang_char/input.decode
ec5a9b24acc35469229e41256ceaf77d  data/lang_char/input.txt
124bf3fe7ce3b73b1994234c15268577  data/lang_char/train_unigram100.model
0df2488cc8eaace95eb12713facb5cf0  data/lang_char/train_unigram100_units.txt
46360cac35c751310e8e8ffd3a034cb5  data/lang_char/train_unigram100.vocab
```

```
==> data/lang_char/input.txt <==
mister quilter is the apostle of the middle classes and we are glad to welcome his gospel
nor is mister quilter's manner less interesting than his matter
he tells us that at this festive season of the year with christmas and roast beef looming before us similes drawn from eating and its results occur most readily to the mind
he has grave doubts whether sir frederick leighton's work is really greek after all and can discover in it but little of rocky ithaca
linnell's pictures are a sort of up guards and at em paintings and mason's exquisite idylls are as national as a jingo poem mister birket foster's landscapes smile at one much in the same way that mister carker used to flash his teeth and mister john collier gives his sitter a cheerful slap on the back before he says like a shampooer in a turkish bath next man
it is obviously unnecessary for us to point out how luminous these criticisms are how delicate in expression
on the general principles of art mister quilter writes with equal lucidity
painting he tells us is of a different quality to mathematics and finish in art is adding more fact
as for etchings they are of two kinds british and foreign
he laments most bitterly the divorce that has been made between decorative art and what we usually call pictures makes the customary appeal to the last judgment and reminds us that in the great days of art michael angelo was the furnishing upholsterer

==> data/lang_char/input.bpe <==
▁mi ster ▁quilter ▁ is ▁the ▁a p ost le ▁o f ▁the ▁mi d d le ▁c las s es ▁ and ▁we ▁ar e ▁g l a d ▁ to ▁we l c om e ▁h is ▁g o s pe l
▁ n or ▁ is ▁mi ster ▁quilter ' s ▁ma nne r ▁ l ess ▁in ter es t ing ▁tha n ▁h is ▁ma t ter
▁h e ▁ t e ll s ▁us ▁tha t ▁ at ▁ t h is ▁f es t ive ▁ s e ason ▁o f ▁the ▁ y e ar ▁w ith ▁ ch r is t m a s ▁ and ▁ro a s t ▁be e f ▁ l o om ing ▁be fore ▁us ▁ s i mile s ▁d r a w n ▁f r om ▁ e at ing ▁ and ▁it s ▁re s u l t s ▁o c c ur ▁m ost ▁re a di l y ▁ to ▁the ▁ mind
▁h e ▁ ha s ▁g r a v e ▁d o u b t s ▁w h e t h er ▁ s i r ▁f r e d er ic k ▁ l eig h to n ' s ▁w or k ▁ is ▁re all y ▁gre e k ▁a f ter ▁ all ▁ and ▁c a n ▁di s c o v er ▁in ▁it ▁b u t ▁li t t le ▁o f ▁ro ck y ▁it ha c a
▁li nne ll ' s ▁ p ic tur es ▁ar e ▁a ▁ s or t ▁o f ▁ u p ▁g u ar d s ▁ and ▁ at ▁ em ▁painting s ▁ and ▁m ason ' s ▁ e x q u is i t e ▁ i d y ll s ▁ar e ▁a s ▁ n at ion a l ▁a s ▁a ▁ j ing o ▁ p o em ▁mi ster ▁b i r k e t ▁f o ster ' s ▁ l and s c a pe s ▁ s mile ▁ at ▁on e ▁m u ch ▁in ▁the ▁ s a m e ▁w a y ▁tha t ▁mi ster ▁c ar k er ▁us e d ▁ to ▁f las h ▁h is ▁ t e e t h ▁ and ▁mi ster ▁ j o h n ▁c o ll i er ▁g ive s ▁h is ▁ s i t ter ▁a ▁ ch e er f u l ▁ s l a p ▁on ▁the ▁b a ck ▁be fore ▁h
e ▁ s a y s ▁li k e ▁a ▁ s ha m p o o er ▁in ▁a ▁ tur k is h ▁b at h ▁ n e x t ▁ma n
▁it ▁ is ▁o b v i o u s l y ▁ u nne c ess ar y ▁for ▁us ▁ to ▁ p o i n t ▁o u t ▁h o w ▁ l u m i n o u s ▁the s e ▁c rit ic is m s ▁ar e ▁h o w ▁d e l ic at e ▁in ▁ e x p r ess ion
▁on ▁the ▁g e n er a l ▁ p r i n c i p l es ▁o f ▁ar t ▁mi ster ▁quilter ▁w rit es ▁w ith ▁ e qual ▁ l u c i di t y
▁painting ▁h e ▁ t e ll s ▁us ▁ is ▁o f ▁a ▁di f f er e n t ▁ qual i t y ▁ to ▁ma t h em at ic s ▁ and ▁f i nish ▁in ▁ar t ▁ is ▁a d d ing ▁m or e ▁f a c t
▁a s ▁for ▁ e t ch ing s ▁the y ▁ar e ▁o f ▁ t w o ▁ k i n d s ▁b rit is h ▁ and ▁for eig n
▁h e ▁ l a ment s ▁m ost ▁b i t ter l y ▁the ▁di v or c e ▁tha t ▁ ha s ▁be e n ▁ma d e ▁be t w e e n ▁d e c or at ive ▁ar t ▁ and ▁w ha t ▁we ▁us u all y ▁c all ▁ p ic tur es ▁ma k es ▁the ▁c u s t om ar y ▁a p pe a l ▁ to ▁the ▁ las t ▁ j u d g ment ▁ and ▁re mind s ▁us ▁tha t ▁in ▁the ▁gre at ▁d a y s ▁o f ▁ar t ▁mi c ha e l ▁a n g e l o ▁w a s ▁the ▁f ur nish ing ▁ u p h o l ster er

==> data/lang_char/input.decode <==
mister quilter is the apostle of the middle classes and we are glad to welcome his gospel
nor is mister quilter's manner less interesting than his matter
he tells us that at this festive season of the year with christmas and roast beef looming before us similes drawn from eating and its results occur most readily to the mind
he has grave doubts whether sir frederick leighton's work is really greek after all and can discover in it but little of rocky ithaca
linnell's pictures are a sort of up guards and at em paintings and mason's exquisite idylls are as national as a jingo poem mister birket foster's landscapes smile at one much in the same way that mister carker used to flash his teeth and mister john collier gives his sitter a cheerful slap on the back before he says like a shampooer in a turkish bath next man
it is obviously unnecessary for us to point out how luminous these criticisms are how delicate in expression
on the general principles of art mister quilter writes with equal lucidity
painting he tells us is of a different quality to mathematics and finish in art is adding more fact
as for etchings they are of two kinds british and foreign
he laments most bitterly the divorce that has been made between decorative art and what we usually call pictures makes the customary appeal to the last judgment and reminds us that in the great days of art michael angelo was the furnishing upholsterer


==> data/lang_char/train_unigram100_units.txt <==
<blank> 0
<unk> 1
' 2
a 3
all 4
and 5
ar 6
ason 7
at 8
b 9

==> data/lang_char/train_unigram100.vocab <==
<unk>   0
<s>     0
</s>    0
▁       -2.01742
e       -2.7203
s       -2.82989
t       -2.99689
l       -3.53267
n       -3.84935
o       -3.88229
```
