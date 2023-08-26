import pickle
import random
# from common.helper import *
from benchmarkOption import *
# For dummy feature 
FEATURE_USED ={
    "linnerud": [0, 1, 2],
    "cancer":[0,1,2],
    "wine":[0,1,2],
    "digits-10":[0,1,2],
    "digits-12":[0,1,2],
    "digits-15":[0,1,2],
    "diabets-18":[0,1,2],
    "diabets-20":[0,1,2]
}

class Node:
    def __init__(self,pNode,threshold,leafValue,feature):
        self.left = None
        self.right = None
        self.isLeaf = False
        self.classId=None
        self.height=0
        self.featureIdx = feature

        self.parent = pNode
        if self.parent is not None:
            self.height = self.parent.getHeight()+1

        self.threshold = threshold
        self.condition = 1 #Denotes lessThan operation or equal operation
        self.leafValue = leafValue

    def setAsLeaf(self):
        self.isLeaf = True
        for i, v in enumerate(self.leafValue):
            if v != 0:
                self.classId = i
                break

    def getHeight(self):
        return self.height

# print(clf.tree_.threshold)
# print(clf.tree_.value)
# totalNum = 2**(clf.get_depth()+1) -1
class BinaryTree:
    def __init__(self, name):
        clf=None
        self.modelName = name
        # with open("data/train/"+name+".model", 'rb') as fid:
        filePath = "/root/PDTE/" if BENCHMARK_OVER_CLOUD else "/home/crypto/Desktop/PDTE/"
        # filePath = "/root/PDTE/"
        with open(filePath+"data/train/"+name+".model", 'rb') as fid:
            clf=  pickle.load(fid)

        self.thresholds = clf.tree_.threshold
        self.values = clf.tree_.value
        self.features = clf.tree_.feature
        self.dummyLeafCnt = 0
        
        # print("binary tree features are: ", self.features)
        # self.featureUsed = clf.n_features_in_int
        print("nodes num is: ",len(self.thresholds))
        # print(self.values)
        
        # count used feature number here
        usedFeatures={}
        cnt=0
        for f in self.features:
            if f != -2:
                if str(f) not in usedFeatures.keys():
                    usedFeatures[str(f)] = cnt
                    cnt+=1
        self.usedFeatureLen = len(usedFeatures.keys())
        # print("#Used_features is: ", self.usedFeatureLen)
        # print(usedFeatures)

        self.usedFeaturesMapping = usedFeatures.copy()

        renamedFeatures = []
        # self.features.copy()
        for v in self.features:
            if v== -2:
                renamedFeatures.append(-2)
            else:
                renamedFeatures.append(usedFeatures[str(v)])
        self.features = renamedFeatures
        # print(self.features)


        # print(clf.tree_.feature)
        self.maxHeight = clf.get_depth()
        self.dummyNodesCnt = 2**(self.maxHeight+1)-1-len(self.thresholds)
        # nodesNum = len(thresholds)
        # print(totalNum)
        idx,self.root = self.middleOrderCreate(None,0)

    def getMaxHeight(self):
        return self.maxHeight
    
    def getUsedFeatures(self):
        return self.usedFeatureLen
    
    def getFeatureIdxMapping(self):
        return self.usedFeaturesMapping

    def middleOrderCreate(self,parent,idx):
        newIdx, middleNode = 0,None
        leafValue = self.values[idx][0]
        leafValue = [int(v) for v in leafValue]
        # print("leafValue is: ",leafValue)

        threshold = self.thresholds[idx]
        if threshold == -2:
            rndIndex = random.choice(FEATURE_USED[self.modelName])
            newIdx, middleNode = idx + \
                1, Node(parent, threshold, leafValue, rndIndex)
            # print("leafValue: ",leafValue)
            # self.dummyNodesCnt -= 1
            self.addDummyNode(middleNode,leafValue)
        else:
            newIdx, middleNode = idx + \
                1, Node(parent, threshold, leafValue, self.features[idx])
            newIdx,leftNode = self.middleOrderCreate(middleNode,newIdx)
            newIdx,rightNode = self.middleOrderCreate(middleNode,newIdx)
            middleNode.left =leftNode
            middleNode.right =rightNode
        return newIdx,middleNode

    def addDummyNode(self,parent,leafValue):
        # self.dummyNodesCnt += 1
        if parent.getHeight() < self.getMaxHeight(): # Automatically insert dummy nodes with nonleaf node
            
            leftNode = Node(parent, -2, leafValue,
                            random.choice(FEATURE_USED[self.modelName]))
            rightNode = Node(parent, -2, leafValue,
                             random.choice(FEATURE_USED[self.modelName]))
            parent.left = leftNode
            parent.right = rightNode
            self.addDummyNode(leftNode,leafValue)
            self.addDummyNode(rightNode,leafValue)
        else:
            # self.dummyNodesCnt -= 1
            # self.dummyLeafCnt += 1
            parent.setAsLeaf()
    
    def getNodesInfo(self,CONVERT_FACTOR):
        print(self.modelName," is with ",self.dummyNodesCnt," dummy nodes.")
        print(self.modelName, " is with ", self.maxHeight, " depth.")


        queue = [self.root]
        nonLeafNodes=[]
        leafNodes=[]
        while len(queue) > 0:
            top = queue.pop(0)
            if top.isLeaf:
                leafNodes.append(top.classId)
            else:
                queue.append(top.left)
                queue.append(top.right)
                t = int(top.threshold*CONVERT_FACTOR)
                # if t == -200:
                #     t=1
                tuple = (top.featureIdx,t )
                nonLeafNodes.append(tuple)
        return leafNodes, nonLeafNodes
        # print("nonleaf len is: ", len(nonLeafNodes))
        # print("leaf len is: ", len(leafNodes))
        # print(leafNodes)
        # print(nonLeafNodes)
