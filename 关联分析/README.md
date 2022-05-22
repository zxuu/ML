# 关联分析
<img src="https://github.com/zxuu/ML/blob/main/images/rela_anal1.png">
结果
项数 1 : [frozenset({5}), frozenset({2}), frozenset({3}), frozenset({1})]

项数 2 : [frozenset({2, 3}), frozenset({3, 5}), frozenset({2, 5}), frozenset({1, 3})]

项数 3 : [frozenset({2, 3, 5})]

项数 4 : []

frozenset({3}) --> frozenset({2}) conf: 0.6666666666666666

frozenset({2}) --> frozenset({3}) conf: 0.6666666666666666

frozenset({5}) --> frozenset({3}) conf: 0.6666666666666666

frozenset({3}) --> frozenset({5}) conf: 0.6666666666666666

frozenset({5}) --> frozenset({2}) conf: 1.0

frozenset({2}) --> frozenset({5}) conf: 1.0

frozenset({3}) --> frozenset({1}) conf: 0.6666666666666666

frozenset({1}) --> frozenset({3}) conf: 1.0

frozenset({3, 5}) --> frozenset({2}) conf: 1.0

frozenset({2, 5}) --> frozenset({3}) conf: 0.6666666666666666

frozenset({2, 3}) --> frozenset({5}) conf: 1.0

frozenset({3, 5}) --> frozenset({2}) conf: 1.0

frozenset({2, 5}) --> frozenset({3}) conf: 0.6666666666666666

frozenset({2, 3}) --> frozenset({5}) conf: 1.0
