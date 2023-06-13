from common.helper import *
from common.constants import *
import random

class EvalParty:
    def __init__(self, id, dimension, seed=None):
        #Basic variables
        self.ID = id
        self.dim = dimension

        if seed is not None:
            offsetChar = chr(ord('a') + seed)
            # self.prg = PRF(
            #     (SEED_KEYS[id][0]+SEED_KEYS[id][1]+offsetChar).encode('ascii'), VEC_VAL_MAX_BOUND)
            # shared PRG with left neighbour
            self.prg1 = PRF(
                (SEED_KEYS[id][0]+offsetChar).encode('ascii'), VEC_VAL_MAX_BOUND)
            # shared PRG with right neighbour
            self.prg2 = PRF(
                (SEED_KEYS[id][1]+offsetChar).encode('ascii'), VEC_VAL_MAX_BOUND)

    def genReplicatedVecShare(self, msg):
        a1 = self.prg1(msg.encode('ascii'), self.dim)
        a2 = self.prg2(msg.encode('ascii'), self.dim)
        return [a1,a2]

    def genZeroShare(self,msg,amount,bound):
        a1 = self.prg1( msg.encode('ascii'), amount)
        a2 = self.prg2( msg.encode('ascii'), amount)
        ret = vec_sub_withBound(a1,a2,bound)
        return ret