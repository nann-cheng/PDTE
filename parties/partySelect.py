from common.helper import *
from common.constants import *
# import statistics

#defines different different actions for different phases
class SelectParty:
    def __init__(self,id,dimension,seed=None):
        #Basic variables
        self.ID=id
        self.dim = dimension


        if seed is not None:
            offsetChar = chr(ord('a') + seed)
            self.prg = PRF(
                (SEED_KEYS[id][0]+SEED_KEYS[id][1]+offsetChar).encode('ascii'), VEC_VAL_MAX_BOUND)
            # shared PRG with left neighbour
            self.prg1 = PRF(
                (SEED_KEYS[id][0]+offsetChar).encode('ascii'), VEC_VAL_MAX_BOUND)
            # shared PRG with right neighbour
            self.prg2 = PRF(
                (SEED_KEYS[id][1]+offsetChar).encode('ascii'), VEC_VAL_MAX_BOUND)


    def genRndVec(self,msg):
        return self.prg(msg.encode('ascii'), self.dim)

    def get2Of3IntShare(self,msg):
        return [self.prg1(msg.encode('ascii')) % self.dim, 
                self.prg2(msg.encode('ascii')) % self.dim]

    def plainRoulette1(self,input=None):
        if self.ID == 0:
            # generate random vector w (security concern but ignore it)
            xVec = input
            # dimension = len(xVec)
            wVec = self.prg2(b"w_vec", self.dim)
            m1Vec = vec_sub(xVec,wVec)
            return m1Vec
        elif self.ID == 1:
            sNumber=input
            wVec = self.prg1(b"w_vec", self.dim)
            v1Vec = self.prg(b"v1Vec", self.dim)
            m2Vec = vec_sub(circ_shift(wVec, sNumber), v1Vec)
            return [v1Vec,m2Vec]
        return None
        
    def plainRoulette2(self,input=None):
        rVec = vec_sub(self.prg1(b"shareConv", self.dim),
                       self.prg2(b"shareConv", self.dim))

        if self.ID == 1:
            rVec = vec_add(rVec,input)
        elif self.ID == 2:
            m1Vec, m2Vec, sNumber = input[0], input[1], input[2]
            v2Vec = vec_add(circ_shift(m1Vec, sNumber),m2Vec)
            rVec = vec_add(rVec, v2Vec)
        return rVec

    def Roulete1(self, xShare, nShare, vecData):
        # xShare = input[0]
        # nShare = input[1]
        if self.ID == 0:
            aVec = vecData
            # print("a vec s")
            v1Vec = circ_shift( vec_add(xShare[0],xShare[1]), (nShare[0]+nShare[1])%self.dim)
            v1Vec = vec_add(v1Vec, aVec)
            return v1Vec
        elif self.ID == 1:
            bVec = vecData
            v2Vec = circ_shift(xShare[1], (nShare[0]+nShare[1]) % self.dim)
            v2Vec = vec_add(v2Vec, bVec)
            return v2Vec
        return None

    def Roulete2(self, input=None):
        alphaVec = input[0]
        nShare = input[1]
        if self.ID == 0:
            v2Vec = input[2]
            w1Vec = circ_shift(v2Vec,nShare[0])
            return [vec_add(w1Vec, alphaVec[0]), alphaVec[1]]
        elif self.ID == 1:
            v1Vec = input[2]
            w3Vec = circ_shift(v1Vec, nShare[1])
            return [alphaVec[0], vec_add(w3Vec, alphaVec[1])]
        elif self.ID == 2:
            v1Vec = input[2]
            v2Vec = input[3]
            w1Vec = circ_shift(v2Vec, nShare[1])
            w3Vec = circ_shift(v1Vec, nShare[0])
            return [vec_add(w3Vec, alphaVec[0]), vec_add(w1Vec, alphaVec[1])]
        return None
