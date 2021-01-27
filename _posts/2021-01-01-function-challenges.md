---
title: "Solving Python Challenges from edabit"
date: 2021-01-01
tags: [Python]
excerpt: "edabit is a great resource to learn and practice solving function challenges"
mathjax: "true"
---

### Merge Lists in Order
Given two lists, merge them to one list and sort the new list in the same order as the first list.

```python
def merge_sort(lst1, lst2):
    if lst1[0] > lst1[1]:
        return sorted(lst1 + lst2, reverse=True)
    else:
        return sorted(lst1 + lst2)

lst1 = [1, 2, 3]
lst2= [5, 4, 6]
merge_sort(lst1, lst2)
```
### Burglary Series (03): Is It Gone?
Your spouse is not concerned with the loss of material possessions but rather with his/her favorite pet. Is it gone?!

Given a dictionary of the stolen items and a string in lower cases representing the name of the pet (e.g. "rambo"), return:

"Rambo is gone..." if the name is on the list.
"Rambo is here!" if the name is not on the list.

```python
def find_it(items, name):
    if name in items.keys():
        return '{} is gone...'.format(name.capitalize())
    return '{} is here!'.format(name.capitalize())

find_it({"tv": 30, "stereo": 50,}, "rocky")
```

### Designing Rugs
Write a function that accepts the width and height (m, n) and an optional proc s and generates a list with m elements.

```python
def make_rug(m, n, s='#'):
    return [s * n] * m

make_rug(3, 5, '$')
```

### Stock Picker
Create a function that takes a list of integers that represent the amount in dollars that a single stock is worth, and return the maximum profit that could have been made by buying stock on day `x` and selling stock on day `y` where `y>x`.

```python
def stock_picker(lst):
    profit_lst = []
    for i, v in enumerate(lst):
        if i + 1 < len(lst):
            profit_lst.append(max(lst[i+1:]) - v)
        if max(profit_lst) > 0:
            return max(profit_lst)
    else:
        return -1
    
stock_picker([1, 2, 4, 10, 100, 2, 3])
```
