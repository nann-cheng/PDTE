from tno.mpc.communication import Pool
import sys
import time
import sys
import asyncio

from dealer import Dealer
from common.constants import *
from common.helper import *

from benchmarkOption import *

from parties.partySelect import SelectParty
from parties.partyEval import EvalParty
from bTree import BinaryTree
from sklearn.datasets import load_wine, load_breast_cancer, load_digits, load_linnerud, load_diabetes


# This's only for the purpose of benchmarking offline cost~
# Not requiring correctness, but only benchmarking purpose.
class Player:
    def __init__(self,id,networkPool):
        self.ID = id
        self.network = networkPool
        self.resetMsgPoolAsList()

        self.prg1 = PRF(
                (SEED_KEYS[id][0]+"randomSeed").encode('ascii'), VEC_VAL_MAX_BOUND)
        # shared PRG with right neighbour
        self.prg2 = PRF(
            (SEED_KEYS[id][1]+"randomSeed").encode('ascii'), VEC_VAL_MAX_BOUND)
        
        self.selectPartySeed=23
    
    def resetMsgPoolAsList(self):
        self.msgPool={
            str( (self.ID+1)%3 ):[],
            str( (self.ID+2)%3 ):[]
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
    
    def getZeroShares(self,msg,bound):
        a1 = self.prg1(msg.encode('ascii'), 1)[0]%bound
        a2 = self.prg2(msg.encode('ascii'), 1)[0]%bound
        return (a1-a2+bound)%bound

    def getNodesAmount(self):
        return self.nodesAmount

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

    # A simple simulation of the offline protocol:
    def Rouletteprep0(self):
        # seed = random.randint(0, 25)
        if self.ID ==0:
            p0 = SelectParty(0, self.tVecDim, self.selectPartySeed)  # i
            aVec = p0.genRndVec("aVec")
            self.msgPool["2"].append( [aVec]*self.nodesAmount )
        elif self.ID==1:
            p1 = SelectParty(1, self.tVecDim, self.selectPartySeed)  # j
            bVec = p1.genRndVec("bVec")
            self.msgPool["2"].append( [bVec]*self.nodesAmount )
    
    # A simple simulation of the offline protocol:
    def Rouletteprep1(self):
        # seed = random.randint(0, 25)
        if self.ID ==0:
            p0 = SelectParty(0, self.tVecDim, self.selectPartySeed)  # i
            aVec = p0.genRndVec("aVec")
            self.msgPool["1"].append( [aVec]*self.nodesAmount )
        elif self.ID==1:
            p1 = SelectParty(1, self.tVecDim, self.selectPartySeed)  # j
            bVec = p1.genRndVec("bVec")
            self.msgPool["2"].append( [bVec]*self.nodesAmount )
        else:
            p1 = SelectParty(1, self.tVecDim, self.selectPartySeed)  # j
            bVec = p1.genRndVec("bVec")
            self.msgPool["0"].append( [bVec]*self.nodesAmount )


    def PathEvalPrep0(self,dim):
        # seed = random.randint(0, 25)
        if self.ID==1:
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

            self.msgPool["2"].append( mVec )
    

    def PathEvalPrep1(self,dim):
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

        self.msgPool[ str( (self.ID+1)%3 )].append( mVec )

        
    

async def async_main(_id):
    # Create the network pool for the current server
    pool = Pool()
    pool.add_http_server( addr=SERVER_IPS[_id], port=NEWTWORK_PORTS[_id] )
    pool.add_http_client("server"+ str( (_id+1)%3 ), addr=SERVER_IPS[(_id+1)%3], port=NEWTWORK_PORTS[(_id+1)%3])
    pool.add_http_client("server"+ str( (_id+2)%3 ), addr=SERVER_IPS[(_id+2)%3], port=NEWTWORK_PORTS[(_id+2)%3])

    ### <<<<< 0. Setup phase ###
    dealer = Dealer()
    leafShares,sharesSampleVec,treeShares,condShare = dealer.getInputData(_id)
    # sharesVec: the sample vector to test
    
    dim = len(treeShares[0])
    
    # print("The amount of nonLeaf nodes are: ", len(bShares))
    
    ### 0. Setup phase >>>>> ###
    player = Player(_id,pool)
    player.inputRSS(leafShares,sharesSampleVec,treeShares,condShare)

    # Setup the network connection 
    if _id==0:
        for i in range(2):
            serverName = "server"+str(1+i)
            message = await pool.recv(serverName)
            # print(serverName,":",message)
    elif _id==1:
        message = await pool.recv("server2")
        # print("server2: "+message)
        await pool.send("server0","Hello!")
    else:
        await pool.send("server1","Hello!")
        await pool.send("server0","Hello!")


    nowRecv = pool.getRecvBytes()
    lastRecv = nowRecv

    ## Test offline cost (commu. & computation)
    # player.Rouletteprep0()
    # await player.distributeNetworkPool()
    # if _id == 2:
    #     msg0 = await pool.recv("server1" )
    #     msg1 =  await pool.recv("server0")
    # player.resetMsgPoolAsList()#Clear message pool
    # player.Rouletteprep1()
    # await player.distributeNetworkPool()

    # if _id == 0:
    #     msg = await pool.recv("server2" )
    # elif _id == 1:
    #     msg = await pool.recv("server0" )
    # else:
    #     msg = await pool.recv("server1" )
    # player.resetMsgPoolAsList()#Clear message pool
    # print("1st Round-Rouletteprep completed")
    ### 1. Feature selection >>>>> ###






    # Test offline cost (commu. & computation)
    player.PathEvalPrep0(dim)
    await player.distributeNetworkPool()
    if _id == 2:
        msg0 = await pool.recv("server1" )
    player.resetMsgPoolAsList()#Clear message pool
    player.PathEvalPrep1(dim)
    await player.distributeNetworkPool()

    if _id == 0:
        msg = await pool.recv("server2" )
    elif _id == 1:
        msg = await pool.recv("server0" )
    else:
        msg = await pool.recv("server1" )
    player.resetMsgPoolAsList()#Clear message pool
    print("1st Round-PathEvalprep completed")
    ### 3. Path evaluation prep >>>>> ###


    



    nowRecv = pool.getRecvBytes()
    print(nowRecv-lastRecv)
    lastRecv = nowRecv
    await pool.shutdown()

if __name__ == "__main__":
    _id = int(sys.argv[1])
    # _id = 0
    loop = asyncio.get_event_loop()
    loop.run_until_complete(async_main(_id))