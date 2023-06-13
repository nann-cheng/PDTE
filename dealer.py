import numpy as np
import sycret
import random
from common.helper import *
from parties.partySelect import SelectParty
from parties.partyEval import EvalParty
from fss import ICNew
from bTree import BinaryTree
from sklearn.datasets import load_wine, load_breast_cancer, load_digits, load_linnerud, load_diabetes
from benchmarkOption import *

class Dealer:
    def __init__(self,):
        self.eq = sycret.EqFactory(n_threads=6)

        
        # self.le = sycret.LeFactory(n_threads=6)
        self.IntCmp = ICNew()
        self.tVecDim = 0
        self.selectPartySeed=23

    def PlainRoulette(self, xVec, s2, s3):
        # seed = random.randint(0,25)
        seed = self.selectPartySeed
        dim=len(xVec)
        p0 = SelectParty(0, dim, seed)  # i
        p1 = SelectParty(1, dim, seed)  # j
        p2 = SelectParty(2, dim, seed)  # k

        m1Vec = p0.plainRoulette1(xVec)

        # statistics.calcCommu(dim, "Select", "offline")

        tmp = p1.plainRoulette1(s2)
        v1Vec, m2Vec = tmp[0], tmp[1]

        # statistics.calcCommu(dim, "Select", "offline")
        # print(m2Vec)
        r1 = p0.plainRoulette2()
        r2 = p1.plainRoulette2(v1Vec)
        r3 = p2.plainRoulette2([m1Vec, m2Vec,s3])

        # statistics.calcCommu(dim, "Select", "offline")
        # statistics.calcCommu(dim, "Select", "offline")
        # statistics.calcCommu(dim, "Select", "offline")

        #Share inverse conversion locally
        return [[r1, r2], [r2, r3], [r3, r1]]

    def Rouletteprep(self,_id):
        # seed = random.randint(0, 25)
        p0 = SelectParty(0, self.tVecDim, self.selectPartySeed)  # i
        p1 = SelectParty(1, self.tVecDim, self.selectPartySeed)  # j
        p2 = SelectParty(2, self.tVecDim, self.selectPartySeed) # k

        n0Share = p0.get2Of3IntShare("nRound")
        n1Share = p1.get2Of3IntShare("nRound")
        n2Share = p2.get2Of3IntShare("nRound")

        aVec = p0.genRndVec("aVec")
        bVec = p1.genRndVec("bVec")

        vShare = self.PlainRoulette(aVec, n1Share[1], n2Share[0])
        wShare = self.PlainRoulette(bVec, n2Share[1], n0Share[0])

        alphaShare = []
        for i in range(3):
            alphaShare.append( vec_array_sub(vShare[i], wShare[i]) )

        a = [aVec, alphaShare[0], n0Share]
        b = [bVec, alphaShare[1], n1Share]
        c = [0,    alphaShare[2], n2Share]
        auxs = [a,b,c]
        # print("auxs is: ",auxs)
        # print("aVec is: ",aVec)
        # print("alphaShare[0] is: ",alphaShare[0])
        # print("n0Share is: ",n0Share)
        # print("n1Share is: ",n1Share)
        return auxs[_id]

    def PathEvalPrep(self,dim,_id):
        # seed = random.randint(0, 25)
        seed = 0
        # dim = 2**height-1
        p1 = EvalParty(0, dim, seed)
        p2 = EvalParty(1, dim, seed)
        p3 = EvalParty(2, dim, seed)

        tao1 = p1.genReplicatedVecShare("permuRand")[0]
        tao1 = [v % BOOLEAN_BOUND for v in tao1]
        tao2 = p2.genReplicatedVecShare("permuRand")[0]
        tao2 = [v % BOOLEAN_BOUND for v in tao2]
        tao3 = p3.genReplicatedVecShare("permuRand")[0]
        tao3 = [v % BOOLEAN_BOUND for v in tao3]

        sVec = p1.prg2("rndS".encode('ascii'), dim)
        sVec = p2.prg1("rndS".encode('ascii'), dim)
        sVec = [v % BOOLEAN_BOUND for v in sVec]

        # p1 locally computes mu1
        invTao1 = inversePermutation(tao1)
        mu1 = shuffleNonLeaf(sVec, invTao1)
        tmp = shuffleNonLeaf(tao1, invTao1)
        mu1 = vec_add_withBound(mu1,tmp,BOOLEAN_BOUND)

        tmp = shuffleNonLeaf(tao2, inversePermutation(tao2))
        tmp = shuffleNonLeaf(tmp, invTao1)
        mu1 = vec_add_withBound(mu1, tmp, BOOLEAN_BOUND)

        # p2 locally computes m
        tmp = shuffleNonLeaf(tao3, inversePermutation(tao3))
        tmp = shuffleNonLeaf(tmp, inversePermutation(tao2))
        mVec = vec_add_withBound(tmp, sVec, BOOLEAN_BOUND)

        # statistics.calcCommu(len(mVec)/32, "Evaluate", "offline")

        # p3 locally computes u3
        mu3 = shuffleNonLeaf(mVec, inversePermutation(tao1))

        # u1,u3 to replicated sharing
        piShares = [[tao1, tao2], [tao2, tao3], [tao3, tao1]]

        # print("tao1 is: ", tao1)
        # print("tao2 is: ", tao2)
        # print("tao3 is: ", tao3)

        # bVal = vec_add_withBound(mu1, mu3, BOOLEAN_BOUND)
        # print("bVal is: ",bVal)
        # statistics.calcCommu(len(mVec)*3/32, "Evaluate", "offline")

        bShares = toReplicatedConverse(mu1,mu3,BOOLEAN_BOUND)
        return (piShares[_id],bShares[_id])

    def distributeBeaverTriple(self):
        allVals = [[] for i in range(3)]
        for i in range(self.capacity):
            alpha = random.randint(0,1)
            beta = random.randint(0,1)
            alphaShares = int_share(alpha, BOOLEAN_BOUND)
            betaShares = int_share(beta, BOOLEAN_BOUND)
            gammaShares = int_share(alpha*beta, BOOLEAN_BOUND)
            for j in range(3):
                allVals[j].append([alphaShares[j], betaShares[j], gammaShares[j] ])
        return allVals

    def distributeCmpKeys(self):
        fssKey0List=[]
        fssKey1List=[]
        for i in range(self.capacity):
            # Equality keys
            eq_keys_a, eq_keys_b = self.eq.keygen(1)
            alpha = self.eq.alpha(eq_keys_a, eq_keys_b)
            alpha = alpha[0].item()
            # print("eq alpha is: ",alpha)
            alpha_shares = int_2of2_share(alpha,INT_32_MAX)
            e_rin_1 = alpha_shares[0]
            e_rin_2 = alpha_shares[1]

            # LessThan keys
            le_rin_1,le_rin_2,le_k0,le_k1 = self.IntCmp.keyGen()

            keys1 = [eq_keys_a, e_rin_1, le_k0, le_rin_1]
            keys2 = [eq_keys_b, e_rin_2, le_k1, le_rin_2]

            fssKey0List.append(keys1)
            fssKey1List.append(keys2)
        return [fssKey0List,fssKey1List]
    
    def getInputData(self,_id):
        ### Benchmarking controling parameters ###
        choiceData = BENCHMARK_CHOICE_DATASET_IDX
        CONVERT_FACTOR = 1000
        ### Benchmarking controling parameters ###

        ### <<<<< 1. Prepare raw test vector & trained model ###
        datasets = ["wine", "linnerud", "cancer", "digits-10", "digits-12", "digits-15","diabets-18","diabets-20"]
        curTree = BinaryTree( datasets[choiceData] )
        leafVector, nonLeafNodes = curTree.getNodesInfo(CONVERT_FACTOR)
        # print("leafVector is: ",leafVector)

        # print("leafVector is: ",leafVector)
        self.capacity = len( nonLeafNodes ) 
        wholeSample = [load_wine().data, load_linnerud().data,
                    load_breast_cancer().data, load_digits().data, load_digits().data, load_digits().data,
                    load_diabetes().data,load_diabetes().data]
        # tVec = random.choice(wholeSample[choiceData])
        print("There are in total #", len(wholeSample[choiceData])," piece of tVecData")
        oldTVec = wholeSample[choiceData][0]
        oldTVec = [int(v*CONVERT_FACTOR) for v in oldTVec]

        # print("oldTVec is: ", oldTVec)
        # print("mapping is: ", curTree.getFeatureIdxMapping())

        # Select only those useful features
        tVec = [0]* curTree.getUsedFeatures()
        for k,v in curTree.getFeatureIdxMapping().items():
            tVec[v] = oldTVec[int(k)]
        print("tVec is: ", tVec)
        print("The test vector has ", len(tVec), "attributes")
        ### 1. Prepare raw test vector & trained model >>>>> ###


        ## <<<<< 2. Prepare RSS of the test vector & trained model ##
        # Fix a random seed such that all players get the same secret sharing of shares
        random.seed(1234)
        leafShares = vec_share(leafVector, VEC_VAL_MAX_BOUND)
        sharesVec = vec_share(tVec, VEC_VAL_MAX_BOUND)
        self.tVecDim = len(tVec)
        treeShares = [ [] for i in range(3)]
        for i in range(3):
            treeShares[i].append([])
            treeShares[i].append([])
        condShares = [[]  for i in range(3) ]
        for i,node in enumerate(nonLeafNodes):
            # if i==0:
            #     print(i, " node is: ",node[0])
            #     print("The first selected feature is: ", tVec[node[0]] )
            idShares = int_share(node[0], self.tVecDim)
            # print("node[1] value is: ", node[1])
            thresholdShares = int_share(node[1], INT_32_MAX)
            condShare = int_share(1, BOOLEAN_BOUND)
            for i in range(3):
                treeShares[i][0].append(idShares[i] )
                treeShares[i][1].append(thresholdShares[i] )
                condShares[i].append(condShare[i] )
        
        return leafShares[_id],sharesVec[_id], treeShares[_id], condShares[_id]
        ### 2. Prepare RSS of the test vector & trained model >>>>> ###