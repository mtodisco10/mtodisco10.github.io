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
