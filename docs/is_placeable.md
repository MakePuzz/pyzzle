# is_placeable メソッドの解説
パズルの盤面に対して，ある単語をある向き（縦か横か），ある位置（行・列）に配置できるかどうかを判断するメソッドである。

## スケルトンパズルとは
スケルトンパズルでは白と黒に彩色された n × n の盤面と，辞書と呼ばれる単語のリストが解答者に与えられる。縦もしくは横に連なる，長さ２以上の極大な白マスの区間をスロットと呼ぶ。解答者は，スロットの長さや既に入っている単語をヒントにし，リストに含まれるすべての単語をすべてのスロットに１対１に割り当てることを求められる。インスタンスの例を以下の図１aに，その解答を図１bに示す。

![図1](images/fig01.png)
図1. パズルの問題例(a)と解答(b)

そして，スケルトンパズルが満たすべき条件は以下の通り：
* 白マスはすべて連結である。
* 黒マスの数や配置に制限はない。
* リストに無い単語は現れない。
* 解は唯一である。

`is_placeable`メソッドは，配置しようとしている単語とその位置・向きが，上記の条件を満たすかどうかを判断する。

## コード全文
まずはコード全文を以下に示し，その後，ブロック毎にコードの解説を記す。
なお，Judgementは列挙型のクラスであり，文字列で数字を表現しているにすぎない。

```python
def is_placeable(self, ori, i, j, word, w_len):
    """
    Returns the word placeability.

    Parameters
    ----------
    ori : int
        Direction of the word (0:Vertical, 1:Horizontal)
    i : int
        Row number of the word
    j : int
        Column number of the word
    word : str
        The word to be checked whether it can be added
    w_len : int
        length of the word

    Returns
    -------
    result : int
        Number of the judgment result

    Notes
    -----
    The result number corresponds to the judgment result
    0. The word can be placed (only succeeded)
    1. The preceding and succeeding cells are already filled
    2. At least one place must cross other words
    3. Not a correct intersection
    4. The same word is in use
    5. The Neighbor cells are filled except at the intersection
    6. US/USA, DOMINICA/DOMINICAN problem
    7. The word overlap with the mask (FancyPuzzle only)
    """

    # If 0 words used, return True
    if self.nwords == 0:
        return Judgement.THE_WORD_CAN_BE_PLACED

    # If the preceding and succeeding cells are already filled
    if ori == 0:
        if i > 0 and self.cell[i - 1, j] != "":
            return Judgement.THE_PRECEDING_AND_SUCCEEDING_CELLS_ARE_ALREADY_FILLED
        if i + w_len < self.height and self.cell[i + w_len, j] != "":
            return Judgement.THE_PRECEDING_AND_SUCCEEDING_CELLS_ARE_ALREADY_FILLED
    if ori == 1:
        if j > 0 and self.cell[i, j - 1] != "":
            return Judgement.THE_PRECEDING_AND_SUCCEEDING_CELLS_ARE_ALREADY_FILLED
        if j + w_len < self.width and self.cell[i, j + w_len] != "":
            return Judgement.THE_PRECEDING_AND_SUCCEEDING_CELLS_ARE_ALREADY_FILLED

    # Get empties
    if ori == 0:
        empties = self.cell[i:i + w_len, j] == ""
    if ori == 1:
        empties = self.cell[i, j:j + w_len] == ""

    # At least one place must cross other words
    if np.all(empties == True):
        return Judgement.AT_LEAST_ONE_PLACE_MUST_CROSS_OTHER_WORDS

    # Judge whether correct intersection
    where = np.where(empties == False)[0]
    if ori == 0:
        j_all = np.full(where.size, j, dtype="int")
        if np.any(self.cell[where + i, j_all] != np.array(list(word))[where]):
            return Judgement.NOT_A_CORRECT_INTERSECTION
    if ori == 1:
        i_all = np.full(where.size, i, dtype="int")
        if np.any(self.cell[i_all, where + j] != np.array(list(word))[where]):
            return Judgement.NOT_A_CORRECT_INTERSECTION

    # If the same word is in use, return False
    if word in self.uwords:
        return Judgement.THE_SAME_WORD_IS_IN_USE

    # If neighbor cells are filled except at the intersection, return False
    where = np.where(empties == True)[0]
    if ori == 0:
        j_all = np.full(where.size, j, dtype="int")
        # Left side
        if j > 0 and np.any(self.cell[where + i, j_all - 1] != ""):
            return Judgement.THE_NEIGHBOR_CELLS_ARE_FILLED_EXCEPT_AT_THE_INTERSECTION
        # Right side
        if j < self.width - 1 and np.any(self.cell[where + i, j_all + 1] != ""):
            return Judgement.THE_NEIGHBOR_CELLS_ARE_FILLED_EXCEPT_AT_THE_INTERSECTION
    if ori == 1:
        i_all = np.full(where.size, i, dtype="int")
        # Upper
        if i > 0 and np.any(self.cell[i_all - 1, where + j] != ""):
            return Judgement.THE_NEIGHBOR_CELLS_ARE_FILLED_EXCEPT_AT_THE_INTERSECTION
        # Lower
        if i < self.height - 1 and np.any(self.cell[i_all + 1, where + j] != ""):
            return Judgement.THE_NEIGHBOR_CELLS_ARE_FILLED_EXCEPT_AT_THE_INTERSECTION

    # US/USA, DOMINICA/DOMINICAN problem
    if ori == 0:
        if np.any(self.enable[i:i + w_len, j] == False) or np.all(empties == False):
            return Judgement.US_USA_DOMINICA_DOMINICAN_PROBLEM
    if ori == 1:
        if np.any(self.enable[i, j:j + w_len] == False) or np.all(empties == False):
            return Judgement.US_USA_DOMINICA_DOMINICAN_PROBLEM

    # Returns True if no conditions are encountered.
    return Judgement.THE_WORD_CAN_BE_PLACED
```

## メソッド宣言
まず，宣言文冒頭について
```python
def is_placeable(self, ori, i, j, word, w_len):
``` 

下記の引数情報に基づき，その単語をパズル上に配置可能かどうかを判断する：

|変数名|クラス|意味|
|-----|-----|---|
|ori  |int  |単語の向き（０：縦，１：横）|
|i    |int  |単語の先頭文字の盤面上の行  |
|j    |int  |単語の先頭文字の盤面上の列  |
|word |str  |配置したい単語の文字列     |
|w_len|int  |`word` の長さ            |

## パズルに既に配置された単語が存在しない場合は配置可能
そのままの意味です。下記のコードにおいて，`self.nwords`は，現在のパズルに入っている単語の個数を意味します。それが`0`であれば，配置可能であると判断し，`Judgement.THE_WORD_CAN_BE_PLACED` を返します。
```python
# If 0 words used, return True
if self.nwords == 0:
    return Judgement.THE_WORD_CAN_BE_PLACED
``` 


## 配置単語の前後に既に文字がある場合は配置不可
配置しようと思う位置の前後に別の文字がある場合，
例として次の５×５のパズルを考えましょう。

| |0|1|2|3|4|
|-|-|-|-|-|-|
|**0**|■|■|■|■|■|
|**1**|■|■|■|■|■|
|**2**|■|■|■|■|■|
|**3**|H|O|G|E|■|
|**4**|■|■|■|■|■|

このパズルに，`ori=0, i=0, j=0, word="FOO"` を置こうとした場合，配置後のパズルは

| |0|1|2|3|4|
|-|-|-|-|-|-|
|**0**|F|■|■|■|■|
|**1**|O|■|■|■|■|
|**2**|O|■|■|■|■|
|**3**|H|O|G|E|■|
|**4**|■|■|■|■|■|

となってしまい，存在しない `FOOH` と言う単語が出現してしまいます。これは，スケルトンパズルの条件「リストに無い単語は現れない。」に反するため，このような場合は配置不可能であると判断し，`Judgement.THE_PRECEDING_AND_SUCCEEDING_CELLS_ARE_ALREADY_FILLED` を返します。

```python
# If the preceding and succeeding cells are already filled
if ori == 0:
    if i > 0 and self.cell[i - 1, j] != "":
        return Judgement.THE_PRECEDING_AND_SUCCEEDING_CELLS_ARE_ALREADY_FILLED
    if i + w_len < self.height and self.cell[i + w_len, j] != "":
        return Judgement.THE_PRECEDING_AND_SUCCEEDING_CELLS_ARE_ALREADY_FILLED
if ori == 1:
    if j > 0 and self.cell[i, j - 1] != "":
        return Judgement.THE_PRECEDING_AND_SUCCEEDING_CELLS_ARE_ALREADY_FILLED
    if j + w_len < self.width and self.cell[i, j + w_len] != "":
        return Judgement.THE_PRECEDING_AND_SUCCEEDING_CELLS_ARE_ALREADY_FILLED
```

## 単語のどこかで他の単語とクロスしていなければならない
スケルトンパズルの条件「白マスは全て連続である」を常に満たすために，配置する単語は必ずどこかで別の単語とクロスしていなければなりません。そのために，まずは単語を配置しようとするマス（4文字なら4マス）の空白位置（`empties`）を調べます。  

例として次の５×５のパズルを考えましょう。

| |0|1|2|3|4|
|-|-|-|-|-|-|
|**0**|■|■|■|■|■|
|**1**|■|■|F|■|■|
|**2**|■|■|U|■|■|
|**3**|H|O|G|E|■|
|**4**|■|■|A|■|■|

このパズルに，`ori=0, i=0, j=4, word="FOO"` を置こうとした場合，`empties` は　`[True, True, True]` という要素数３の配列となります（`True`が空白を表す）。`empties`の要素が全て`True`なら配置不可能であると判断し，`Judgement.AT_LEAST_ONE_PLACE_MUST_CROSS_OTHER_WORDS` を返します。
```python
# At least one place must cross other words
if ori == 0:
    empties = self.cell[i:i + w_len, j] == ""
if ori == 1:
    empties = self.cell[i, j:j + w_len] == ""
if np.all(empties == True):
    return Judgement.AT_LEAST_ONE_PLACE_MUST_CROSS_OTHER_WORDS
```


## クロス部分の文字が同じでなければ配置不可
上のチェックにより，単語の少なくとも１文字はクロスしていることがわかりました。次は，そのクロス部分の文字が全て同じであるかをチェックします。異なる単語でクロスしている場合は配置不可能であると判断し，`Judgement.NOT_A_CORRECT_INTERSECTION` を返します。

```python
# Judge whether correct intersection
where = np.where(empties == False)[0]
if ori == 0:
    j_all = np.full(where.size, j, dtype="int")
    if np.any(self.cell[where + i, j_all] != np.array(list(word))[where]):
        return Judgement.NOT_A_CORRECT_INTERSECTION
if ori == 1:
    i_all = np.full(where.size, i, dtype="int")
    if np.any(self.cell[i_all, where + j] != np.array(list(word))[where]):
        return Judgement.NOT_A_CORRECT_INTERSECTION
```


## もし同じ単語が既に盤面上に配置されている場合は配置不可能
これは，単語の重複を避けるための判定です。なぜここにきてこんな簡単な判定をするのか不思議に思う方もいるでしょう。こんなものは最初に見ても良いはずです。実は，`pyzzle`では統計的な調査に基づき，なるべく早く`return`されるような順番に条件分岐を並べています。その結果，この条件分岐がメソッドの中盤で現れているのです。  
なお，もし同じ単語が既に盤面上に配置されている場合は配置不可能であると判断し，`Judgement.THE_SAME_WORD_IS_IN_USE` を返します。

```python
# If the same word is in use, return False
if word in self.uwords:
    return Judgement.THE_SAME_WORD_IS_IN_USE
```

## クロス部の横以外のマスに文字がある場合は配置不可
例として次の５×５のパズルを考えましょう。

| |0|1|2|3|4|
|-|-|-|-|-|-|
|**0**|■|■|■|■|■|
|**1**|■|■|F|■|■|
|**2**|■|■|U|■|■|
|**3**|H|O|G|E|■|
|**4**|■|■|A|■|■|

このパズルに，`ori=0, i=0, j=1, word="PIYO"` を置こうとした場合，配置後のパズルは

| |0|1|2|3|4|
|-|-|-|-|-|-|
|**0**|■|P|■|■|■|
|**1**|■|I|F|■|■|
|**2**|■|Y|U|■|■|
|**3**|H|O|G|E|■|
|**4**|■|■|A|■|■|

となり，`IF`と`YU`という，辞書にない単語が存在することになってしまい「リストに無い単語は現れない。」の条件に反するため，配置不可能です。

この判定には，クロス部以外の文字の両脇に文字が存在するかをチェックします。つまり，以下のパズルの `●` に一つでも文字が存在した場合に配置不可能であるとみなし，`Judgement.THE_NEIGHBOR_CELLS_ARE_FILLED_EXCEPT_AT_THE_INTERSECTION` を返します。

| |0|1|2|3|4|
|-|-|-|-|-|-|
|**0**|`●`|P|`●`|_|_|
|**1**|`●`|I|`●`|_|_|
|**2**|`●`|Y|`●`|_|_|
|**3**|_|O|_|_|_|
|**4**|_|_|_|_|_|

Pythonコードは次の通り。NumPyのfancy indexingによって可能な限り高速化しています。

```python
# If neighbor cells are filled except at the intersection, return False
where = np.where(empties == True)[0]
if ori == 0:
    j_all = np.full(where.size, j, dtype="int")
    # Left side
    if j > 0 and np.any(self.cell[where + i, j_all - 1] != ""):
        return Judgement.THE_NEIGHBOR_CELLS_ARE_FILLED_EXCEPT_AT_THE_INTERSECTION
    # Right side
    if j < self.width - 1 and np.any(self.cell[where + i, j_all + 1] != ""):
        return Judgement.THE_NEIGHBOR_CELLS_ARE_FILLED_EXCEPT_AT_THE_INTERSECTION
if ori == 1:
    i_all = np.full(where.size, i, dtype="int")
    # Upper
    if i > 0 and np.any(self.cell[i_all - 1, where + j] != ""):
        return Judgement.THE_NEIGHBOR_CELLS_ARE_FILLED_EXCEPT_AT_THE_INTERSECTION
    # Lower
    if i < self.height - 1 and np.any(self.cell[i_all + 1, where + j] != ""):
        return Judgement.THE_NEIGHBOR_CELLS_ARE_FILLED_EXCEPT_AT_THE_INTERSECTION
```

## US/USA, DOMINICA/DOMINICAN 問題
最後の判定です。これまでの条件で一見完璧そうに見えますが，実は落とし穴が存在します。それが，`US/USA 問題`です。  
下記のパズルを例に，US/USA 問題を解説します。

| |0|1|2|3|4|
|-|-|-|-|-|-|
|**0**|■|■|F|■|■|
|**1**|■|■|U|S|■|
|**2**|■|■|G|■|■|
|**3**|■|■|A|■|■|
|**4**|■|■|■|■|■|

ここに，新たな単語「`USA`」を `ori=1, i=1, j=2` に配置できるかどうか考えてみてください。実はこの状況では，これまでに述べた全てのチェックをクリアしてしまいます。すると，`US` の上に`USA`が重なってしまい，さらに，単語リストに `US`と`USA`の両者が含まれることとなります。これは，見た目上 `US` が存在せず「リストに無い単語は現れない。」に反すため配置不可能であると判断し，`Judgement.US_USA_DOMINICA_DOMINICAN_PROBLEM` を返します。

```python
# US/USA, DOMINICA/DOMINICAN problem
if ori == 0:
    if np.any(self.enable[i:i + w_len, j] == False) or np.all(empties == False):
        return Judgement.US_USA_DOMINICA_DOMINICAN_PROBLEM
if ori == 1:
    if np.any(self.enable[i, j:j + w_len] == False) or np.all(empties == False):
        return Judgement.US_USA_DOMINICA_DOMINICAN_PROBLEM
```

## 全てのチェックを抜けた単語を配置可能として判定
全ての関門を突破した単語のみがここに辿り着けるため，そのような単語を配置可能と判断し，`Judgement.THE_WORD_CAN_BE_PLACED` を返します。
```python
# Returns True if no conditions are encountered.
return Judgement.THE_WORD_CAN_BE_PLACED
```

# 参考文献
* 西島善治, 反復局所探索法に基づいたスケルトンパズルの生成アルゴリズム. 小樽商科大学 商学部, 卒業論文, 2017.
