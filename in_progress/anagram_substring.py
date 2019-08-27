# Container with most water

# def maxArea(height) -> int:
#     left = 0
#     right = len(height)-1
#     max_area = 0
#     while left < right - 1: 
#         left_height = height[left]
#         right_height = height[right]

#         lesser_height = left_height
#         if left_height > right_height:
#             lesser_height = right_height

#         area = (right - left) * lesser_height
#         if area > max_area:
#             max_area = area
#         left += 1
#         right -= 1
#     return max_area

# a = [1, 8, 6, 2, 5, 4, 8, 3, 7]
# print(maxArea(a))
            
# Valid Parentheses
# def isValid(s):
#     stack = []
#     pairs = {"(":")", "{":"}", "[":"]"}
#     for i in range(len(s)):
#         if s[i] =="(" or s[i] == "{" or s[i]=="[":
#             stack.append(s[i])
#         else:
#             if len(stack) == 0:
#                 return False
#             top = stack.pop()
#             if not pairs.get(top) ==s[i]:
#                 return False
#     if not len(stack) == 0:
#         return False
#     return True

# print(isValid("{}"))
# print(isValid("(){}[]"))
# print(isValid("(]"))
# print(isValid("([}]"))
# print(isValid("{[]}"))


# Implement MapSum
# insert: given a pair of (string,int) where string is key and int is value. If exists already, override it with new one
# sum: given a string that represents prefix, return sum of all pairs' value whose key starts with the prefix.

def isPrefix(pref, word):
    if len(pref) > len(word):
        return False
    for i in range(len(pref)):
        if not pref[i] == word[i]:
            return False
    return True

class MapSum:
    def __init__(self):
        self.m_map = {}

    def insert(self, key:str, val: int) -> None:
        self.m_map[key] = val
    
    def sum(self, prefix: str) -> int:
        map_sum = 0 
        for key in self.m_map:
            if isPrefix(prefix, key):
                map_sum += self.m_map[key]  
        return map_sum

ms = MapSum
ms.insert("apple",3)
print(ms.sum("ap"))
ms.insert