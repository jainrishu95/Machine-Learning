# class A(object):
#     def go(self):
#         print("go A go!")
#     def stop(self):
#         print("stop A stop!")
#     def pause(self):
#         raise Exception("Not Implemented")
#
# class B(A):
#     def go(self):
#         super(B, self).go()
#         print("go B go!")
#
# class C(A):
#     def go(self):
#         super(C, self).go()
#         print("go C go!")
#     def stop(self):
#         super(C, self).stop()
#         print("stop C stop!")
#
# class D(B,C):
#     def go(self):
#         super(D, self).go()
#         print("go D go!")
#     def stop(self):
#         super(D, self).stop()
#         print("stop D stop!")
#     def pause(self):
#         print("wait D wait!")
#
# class E(B,C): pass
#
# a = A()
# b = B()
# c = C()
# d = D()
# e = E()
#
# # specify output from here onwards
#
# a.go()
# b.go()
# c.go()
# d.go()
# e.go()
#
# a.stop()
# b.stop()
# c.stop()
# d.stop()
# e.stop()
#
# a.pause()
# b.pause()
# c.pause()
# d.pause()
# e.pause()

class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


class Solution:
    def construct(self, nums):
        if len(nums) == 0:
            pass
        else:
            max_ = max(nums)
            index = nums.index(max_)
            root = Node(max_)
            root.left = self.construct(nums[:index])
            root.right = self.construct(nums[index + 1:])

    def constructMaximumBinaryTree(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """
        max_ = max(nums)
        index = nums.index(max_)
        root = Node(max_)
        root.left = self.construct(nums[:index])
        root.right = self.construct(nums[index + 1:])
        return root.val

print(Solution().constructMaximumBinaryTree([3,2,1,6,0,5]))
