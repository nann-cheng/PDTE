from common.helper import *
from common.constants import *
from parties.partySelect import SelectParty
from parties.partyEval import EvalParty
from fss import ICNew
import sycret
import numpy as np

#TODO: 
# 1. generate multiple instances of aux values
class Player:
    def __init__(self,id,aux,networkPool):
        self.ID = id
        self.aux = aux
        self.network = networkPool
        self.resetMsgPoolAsList()
        self.eqFSS = sycret.EqFactory(n_threads=6)
        self.IntCmp = ICNew()

        self.prg1 = PRF(
                (SEED_KEYS[id][0]+"randomSeed").encode('ascii'), VEC_VAL_MAX_BOUND)
        # shared PRG with right neighbour
        self.prg2 = PRF(
            (SEED_KEYS[id][1]+"randomSeed").encode('ascii'), VEC_VAL_MAX_BOUND)
    
    def getZeroShares(self,msg,bound):
        a1 = self.prg1(msg.encode('ascii'), 1)[0]%bound
        a2 = self.prg2(msg.encode('ascii'), 1)[0]%bound
        return (a1-a2+bound)%bound

    def getNodesAmount(self):
        return self.nodesAmount
    
    def resetMsgPoolAsList(self):
        self.msgPool={
            str( (self.ID+1)%3 ):[],
            str( (self.ID+2)%3 ):[]
        }
    
    def resetMsgPoolWithKeys(self,key0,key1):
        self.msgPool={
            str( (self.ID+1)%3 ):
                {
                    key0:[],
                    key1:[]
                }
            ,
            str( (self.ID+2)%3 ):
                {
                    key0:[],
                    key1:[]
                }
        }

    async def distributeNetworkPool(self):
        for key,val in self.msgPool.items():
            _type = type(val).__name__
            willSend=False
            if _type == "dict":
                for inKey,inVal in val.items():
                    if len(inVal)>0:
                        willSend=True
            elif _type == "list":
                if len(val)>0:
                    willSend=True
            if willSend:
                await self.network.send("server"+key,val)

    def inputRSS(self,leafSS, vSS, treeSS, condShare):
        self.leafSS = leafSS
        self.leafNodesAmount = len(self.leafSS[0])

        self.vSS = vSS
        self.tVecDim = len(self.vSS[0])
        self.selectP = SelectParty(self.ID, self.tVecDim)
        self.idxSS = treeSS[0]
        self.thresholdSS = treeSS[1]
        self.condShares = condShare
        self.nodesAmount = len(condShare)

        self.selectedShares=[]
        self.vals4Equal = []
        self.vals4Less = []

        self.valsPQ_shares=[]
        self.cmpResultShares=[]

        
        self.shuffledShares=[]# Held by each party, a list of size 2 containing the RSS shares of a shuffled vector
        self.shuffleReveal=0

    def inputFSSKeys(self,fssKeys):
        self.fssKeys = fssKeys

    def inputBeaverTriples(self,triples):
        self.triples = triples

    def featureSelect0(self):
        for i,idx in enumerate(self.idxSS):
            nSS = self.aux[2]
            xorShare = RSS_local_add(idx,nSS,self.tVecDim)
            # if i==0:
            #     print("idx is: ", idx)
            #     print("nSS is: ", nSS)
            #     print("xORshare is: ", xorShare)
            idx_maskSS = xorShare[0]
            #Send out all msgs in one round
            if self.ID <= 1:
                theVec = self.aux[0]
                vVec = self.selectP.Roulete1( self.vSS, nSS, theVec )
                #player 0 send vVec to 1,2
                #player 1 send vVec to 0,2
                #player 0,1,2 send idx_maskSS to 1,2,0
                if self.ID == 0:
                    self.msgPool["1"].append( (vVec,idx_maskSS) )
                    self.msgPool["2"].append( vVec )
                else:
                    self.msgPool["2"].append( (vVec,idx_maskSS) )
                    self.msgPool["0"].append( vVec )
            else:
                self.msgPool["0"].append( idx_maskSS )

    #Process feature selection after network interaction
    def featureSelect1(self,p0,p1):
        nSS = self.aux[2]
        alpha_share = self.aux[1]
        for i,idx in enumerate(self.idxSS):
            m_mask_nSS = RSS_local_add(idx,nSS,self.tVecDim)
            revealShift = m_mask_nSS[0] + m_mask_nSS[1]
            if self.ID == 0:
                # p0,p1: vArray,nSSArray
                yShare = self.selectP.Roulete2([alpha_share, nSS, p0[i] ])
                revealShift = ( revealShift+ p1[i])%self.tVecDim
                self.selectedShares.append([ yShare[0][revealShift],  yShare[1][revealShift]])
            elif self.ID == 1:
                # p0: array:[vArray,nssArray]
                yShare = self.selectP.Roulete2([alpha_share, nSS, p0[i][0] ])
                revealShift = (revealShift + p0[i][1])%self.tVecDim
                self.selectedShares.append([ yShare[0][revealShift],  yShare[1][revealShift]])
            else:
                # p0,p1: vArray,array
                yShare = self.selectP.Roulete2([alpha_share, nSS, p0[i], p1[i][0] ])
                revealShift = (revealShift + p1[i][1])%self.tVecDim
                self.selectedShares.append([ yShare[0][revealShift],  yShare[1][revealShift]])
            # if i==0:
                # print("revealShift is: ", revealShift)
        # print("The first selected feature is: ", self.selectedShares[0] )

    # Collect messages to be sent for FSS evaluation
    def compare0(self):
        if self.ID == 0:
            for i,select in enumerate(self.selectedShares):
                # local convert (2,3)-RSS to (2,2)-SS
                newSelectShare =  inRing(select[0]+select[1], INT_32_MAX)
                newThresholdShare = inRing( self.thresholdSS[i][0] + self.thresholdSS[i][1], INT_32_MAX)
                subShare = inRing( newSelectShare - newThresholdShare, INT_32_MAX)
                self.vals4Equal.append( inRing( subShare + self.fssKeys[i][1], INT_32_MAX) )
                self.vals4Less.append( inRing(  subShare + self.fssKeys[i][3], INT_32_MAX) )
            self.msgPool["1"].append( self.vals4Equal)
            self.msgPool["1"].append( self.vals4Less)
        elif self.ID == 1:
            for i,select in enumerate(self.selectedShares):
                # local convert (2,3)-RSS to (2,2)-SS
                newSelectShare = select[1]
                newThresholdShare = self.thresholdSS[i][1]
                subShare = inRing( newSelectShare - newThresholdShare, INT_32_MAX)
                self.vals4Equal.append( inRing( subShare + self.fssKeys[i][1], INT_32_MAX) )
                self.vals4Less.append( inRing(  subShare + self.fssKeys[i][3], INT_32_MAX) )
            self.msgPool["0"].append( self.vals4Equal)
            self.msgPool["0"].append( self.vals4Less)
        
    # Collect messages to be sent to fulfill SC-AND computation
    def compare1(self,otherShares):
        if self.ID <= 1:
            for i in range(self.nodesAmount):
                reveal =  inRing(self.vals4Equal[i] + otherShares[0][i], INT_32_MAX)
                r_eq = self.eqFSS.eval(self.ID, np.array( [np.int64(reveal)] ), self.fssKeys[i][0])
                r_eq = r_eq[0].item()#Convert numpy array to a normal int value

                reveal =  inRing(self.vals4Less[i]+otherShares[1][i], INT_32_MAX)

                r_le = self.IntCmp.eval(self.ID, np.array( [np.int64(reveal)] ), self.fssKeys[i][2])
                xor = inRing( r_eq + r_le, BOOLEAN_BOUND)

                cmpShare = [inRing( r_eq + self.getZeroShares("bRand",BOOLEAN_BOUND), BOOLEAN_BOUND),0]
                
                if self.ID == 0:
                    condShare = inRing( self.condShares[i][0]+self.condShares[i][1], BOOLEAN_BOUND)
                    alpha = inRing( self.triples[i][0][0]+self.triples[i][0][1], BOOLEAN_BOUND)
                    beta = inRing( self.triples[i][1][0]+self.triples[i][1][1], BOOLEAN_BOUND)
                    m_pVal = inRing( condShare + alpha, BOOLEAN_BOUND)
                    m_qVal = inRing( xor + beta, BOOLEAN_BOUND)

                    self.valsPQ_shares.append( (m_pVal,m_qVal) )
                    self.msgPool["1"]["sc-and"].append( (m_pVal,m_qVal) )
                    self.msgPool["2"]["sc-and"].append( (m_pVal,m_qVal) )
                    self.msgPool["2"]["invConv"].append( cmpShare[0] ) 
                else:
                    condShare = self.condShares[i][1]
                    alpha = self.triples[i][0][1]
                    beta = self.triples[i][1][1]
                    m_pVal = inRing( condShare + alpha, BOOLEAN_BOUND)
                    m_qVal = inRing( xor + beta, BOOLEAN_BOUND)
                    self.valsPQ_shares.append( (m_pVal,m_qVal) )
                    self.msgPool["0"]["sc-and"].append( (m_pVal,m_qVal) )
                    self.msgPool["2"]["sc-and"].append( (m_pVal,m_qVal) )
                    self.msgPool["0"]["invConv"].append( cmpShare[0] ) 

                self.cmpResultShares.append(cmpShare)
        else:
            for i in range(self.nodesAmount):
                cmpShare = [self.getZeroShares("bRand",BOOLEAN_BOUND),0]
                self.msgPool["1"]["invConv"].append( cmpShare[0] )
                self.cmpResultShares.append(cmpShare)
        
    def compare2(self,pq_vals0,pq_vals1,otherBShareList):
        vec=[[],[]]
        for i in range(self.nodesAmount):
            if self.ID <= 1:
                P = inRing( self.valsPQ_shares[i][0] + pq_vals0[i][0], BOOLEAN_BOUND) 
                Q = inRing( self.valsPQ_shares[i][1] + pq_vals0[i][1], BOOLEAN_BOUND)
            else:
                P = inRing( pq_vals0[i][0] + pq_vals1[i][0], BOOLEAN_BOUND)
                Q = inRing( pq_vals0[i][1] + pq_vals1[i][1], BOOLEAN_BOUND)
            cmpShare = self.triples[i][2]
            cmpShare = RSS_local_add(cmpShare, RSS_local_mul( self.triples[i][0],Q, BOOLEAN_BOUND),BOOLEAN_BOUND)
            cmpShare = RSS_local_add(cmpShare, RSS_local_mul( self.triples[i][1],P, BOOLEAN_BOUND),BOOLEAN_BOUND)
            if self.ID == 0:
                cmpShare = RSS_local_add(cmpShare,[0,P*Q], BOOLEAN_BOUND)
            elif self.ID == 1:
                cmpShare = RSS_local_add(cmpShare,[P*Q,0], BOOLEAN_BOUND)
            
            self.cmpResultShares[i][1] =  otherBShareList[i]
            tmp = RSS_local_add(cmpShare,self.cmpResultShares[i], BOOLEAN_BOUND)
            for j in range(2):
                vec[j].append(tmp[j])
        # Reshape self.cmpResultShares from m x 2 to 2 x m
        self.cmpResultShares = vec
    
    # Embed shuffle reveal within this function
    def pathEval0_shuffleReveal(self,bShares,piShares):
        xorShares = []
        for j in range(2):
            xorShares.append( vec_add_withBound(self.cmpResultShares[j], bShares[j], BOOLEAN_BOUND) )
        
        seed = 12
        dim = len(bShares[0])
        party = EvalParty(self.ID, dim, seed)
        alpha = party.genReplicatedVecShare("alphaRnd")
        #shuffleReveal
        if self.ID ==0:
            beta1 = vec_add_withBound(xorShares[0], xorShares[1], BOOLEAN_BOUND)
            pi_1 = piShares[0]
            sigma = vec_sub(shuffleNonLeaf(beta1, pi_1), alpha[0])
            self.msgPool["1"]["shuffleReveal"] =  sigma 
        elif self.ID == 2:
            pi_1 = piShares[1]
            beta3 = xorShares[0]
            gamma = vec_add(shuffleNonLeaf(beta3, pi_1), alpha[1])
            self.msgPool["1"]["shuffleReveal"]=  gamma 

    def pathEval0_optShuffle(self,piShares):
        #seed = random.randint(0, 25)
        seed = 13
        party = EvalParty(self.ID, self.leafNodesAmount, seed)
        alpha = party.genReplicatedVecShare("alphaRnd")
        if self.ID == 0:#P0 to P2
            beta1 = vec_add(self.leafSS[0], self.leafSS[1])
            pi_1 = piShares[0]
            pi_2 = piShares[1]
            sigma = vec_sub(permute(pi_1,beta1),alpha[0])
            sigma = vec_sub(permute(pi_2, sigma), alpha[1])
            self.msgPool["2"]["optShuffle"]= sigma
        elif self.ID == 2:#P2 to P1
            beta3 = self.leafSS[0]
            pi_1 = piShares[1]
            gamma = vec_add(permute(pi_1, beta3),alpha[1])
            self.msgPool["1"]["optShuffle"]= gamma

    # shuffleReveal responded by p1
    def pathEval1_shuffleRevealRespond(self,sigma,gamma,piShares):
        pi_2 = piShares[0]
        pi_3 = piShares[1]
        out = vec_add_withBound(gamma, sigma, BOOLEAN_BOUND)
        out = shuffleNonLeaf(out, pi_2)
        out = shuffleNonLeaf(out, pi_3)
        self.shuffleReveal = list(out)
        self.msgPool["0"]["shuffleReveal"] = list(out)
        self.msgPool["2"]["shuffleReveal"] = list(out)
        # print("out value is:", out)
        # for v in out:
        #     self.msgPool["0"]["shuffleReveal"].append(v)
        #     self.msgPool["2"]["shuffleReveal"].append(v)
        
    def pathEval1_optShuffleRespond(self,gamma,sigma,piShares):
        seed = 13
        party = EvalParty(self.ID, self.leafNodesAmount, seed)
        alpha = party.genReplicatedVecShare("alphaRnd")

        reshapeRnd = party.genZeroShare("randomRnd",self.leafNodesAmount, VEC_VAL_MAX_BOUND)
        if self.ID == 0:
            self.shuffledShares.append( reshapeRnd )
            self.msgPool["2"]["optShuffle"] = reshapeRnd
        elif self.ID == 1:#(P2 to P1:gamma), P1 now Responds
            pi_2 = piShares[0]
            pi_3 = piShares[1]
            out1 = vec_add(permute(pi_2,gamma),alpha[0])
            out1 = permute(pi_3,out1)
            self.shuffledShares.append( vec_add(reshapeRnd,out1) )
            self.msgPool["0"]["optShuffle"] = self.shuffledShares[0]
        elif self.ID == 2:#(P0 to P2:sigma), P2 now Responds
            pi_3 = piShares[0]
            out2 = permute(pi_3, sigma)
            self.shuffledShares.append( vec_add(reshapeRnd,out2) )
            self.msgPool["1"]["optShuffle"] = self.shuffledShares[0]
    
    def pathEval2(self,revealVec,otherShuffleShare):
        self.shuffledShares.append( otherShuffleShare )
        if self.ID != 1:
            self.shuffleReveal = revealVec
        index = getEvalIndex(self.shuffleReveal)
        # print("final index is: ",index)
        v0 = self.shuffledShares[0][index]
        v1 = self.shuffledShares[1][index]
        return v0,v1

