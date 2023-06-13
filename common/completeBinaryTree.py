class BTNode:
    def __init__(self, pNode, value):
        self.left = None
        self.right = None
        self.height = 0
        self.flipValue = value

        self.parent = pNode
        if self.parent is not None:
            self.height = self.parent.getHeight()+1

    def setParentAs(self,pNode):
        self.parent = pNode
        self.height = pNode.getHeight()+1
    
    def setLeftAs(self, leftNode):
        leftNode.setParentAs(self)
        self.left = leftNode
    
    def setRightAs(self, rightNode):
        self.right = rightNode

    def getHeight(self):
        return self.height
    
    def flipLeftRight(self):
        temp = self.left
        self.left = self.right
        self.right = temp

    def getLeft(self):
        return self.left

    def getRight(self):
        return self.right

    def getVal(self):
        return self.flipValue
