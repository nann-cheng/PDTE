from tno.mpc.communication import Pool
import sys
import time
import sys
import asyncio

from dealer import Dealer
from common.constants import *
from common.helper import *

import pickle
from benchmarkOption import *
from player import Player


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
    aux = dealer.Rouletteprep(_id)

    
    
    dim = len(treeShares[0])
    piShares, bShares = dealer.PathEvalPrep(dim,_id)
    # print("The amount of nonLeaf nodes are: ", len(bShares))

    
    allTriples = dealer.distributeBeaverTriple()

    # Since FSS keys are generated by random rust libs, 
    # we need network communication to pass these correlated arrays.

    # if _id == 0:
    # allFssKeys = dealer.distributeCmpKeys()
        

    

    player = Player(_id,aux,pool)
    player.inputRSS(leafShares,sharesSampleVec,treeShares,condShare)
    player.inputBeaverTriples(allTriples[_id])
    
    ### 0. Setup phase >>>>> ###
    
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

    
    # These are mainly for the measurement of the commu bytes in the offline phase
    if BENCHMARK_MEASURE_OFFLINE_COMMU:
        lastRecv = pool.getRecvBytes()
        print("Offline Rouletteprep costs:")
        if _id==0:
            message = [aux]*player.getNodesAmount()
            await pool.send("server1",message)
            print(0)
        elif _id==1:
            message = await pool.recv("server0")
            nowRecv = pool.getRecvBytes()
            print(nowRecv-lastRecv)
            lastRecv = nowRecv
        else:
            print(0)
        
        print("Offline PathEvalPrep costs:")
        if _id==0:
            message = [piShares, bShares]
            await pool.send("server1",message)
            print(0)
        elif _id==1:
            message = await pool.recv("server0")
            nowRecv = pool.getRecvBytes()
            print(nowRecv-lastRecv)
            lastRecv = nowRecv
        else:
            print(0)

        
    if BENCHMARK_MEASURE_OFFLINE_COMMU:
        print("Offline Compare costs:")

    if _id ==0:
        allFssKeys = dealer.distributeCmpKeys()
        player.inputFSSKeys(allFssKeys[0])
        if BENCHMARK_MEASURE_OFFLINE_COMMU:
            await pool.send("server1", [allTriples,pickle.dumps( allFssKeys)]  )
            print(0)
        else:
            await pool.send("server1", pickle.dumps( allFssKeys[1])  )
    elif _id ==1:
        if BENCHMARK_MEASURE_OFFLINE_COMMU:
            message = await pool.recv("server0")
            nowRecv = pool.getRecvBytes()
            print(nowRecv-lastRecv)
            mFssKeysBytes = message[1]
            mFssKeys = pickle.loads(mFssKeysBytes)
            player.inputFSSKeys(mFssKeys[1])
        else:
            mFssKeysBytes = await pool.recv("server0")
            mFssKeys = pickle.loads(mFssKeysBytes)
            player.inputFSSKeys(mFssKeys)
    else:
        if BENCHMARK_MEASURE_OFFLINE_COMMU:
            print(0)


    if BENCHMARK_MEASURE_ONLINE_COMMU:
        lastRecv = pool.getRecvBytes()

    ### <<<<< 1. Feature selection ### (1). Does feature selection on every non-leaf node value
    player.featureSelect0()# Collect First round message
    await player.distributeNetworkPool()

    if _id == 0:
        vArray = await pool.recv("server1" )
        nSSArray =  await pool.recv("server2")
        player.featureSelect1(vArray,nSSArray)
    elif _id == 1:
        array = await pool.recv("server0" )
        player.featureSelect1(array,None)
    else:
        vArray = await pool.recv("server0" )
        array = await pool.recv("server1" )
        player.featureSelect1(vArray,array)
    player.resetMsgPoolAsList()#Clear message pool
    print("1st Round-feature selection completed")
    ### 1. Feature selection >>>>> ###

    if BENCHMARK_MEASURE_ONLINE_COMMU:
        nowRecv = pool.getRecvBytes()
        # print("Player ",_id, " feature selection received: ",nowRecv-lastRecv)
        print(nowRecv-lastRecv)
        lastRecv = nowRecv

    ### <<<<< 2. Comparison phase <Costs 2 rounds> ###
    player.compare0()
    await player.distributeNetworkPool()
    print("2nd Round-comparison: complete eq/less .")

    otherShares=None
    if _id == 0:
        otherShares = await pool.recv("server1" )
    elif _id == 1:
        otherShares = await pool.recv("server0" )
    player.resetMsgPoolWithKeys("sc-and","invConv")
    player.compare1(otherShares)
    await player.distributeNetworkPool()
    print("3rd Round-comparison: complete SC-AND .")

    if BENCHMARK_MEASURE_ONLINE_COMMU:
        nowRecv = pool.getRecvBytes()
        print(nowRecv-lastRecv)
        lastRecv = nowRecv

    if _id == 0:
        messages = await pool.recv("server1" )
        pq_vals = messages["sc-and"]
        otherBShareList = messages["invConv"]
        player.compare2(pq_vals,None,otherBShareList)
    elif _id == 1:
        messages = await pool.recv("server0" )
        pq_vals = messages["sc-and"]
        messages = await pool.recv("server2" )
        otherBShareList = messages["invConv"]
        player.compare2(pq_vals,None,otherBShareList)
    else:
        messages0 = await pool.recv("server0" )
        messages1 = await pool.recv("server1" )
        pq_vals0 = messages0["sc-and"]
        pq_vals1 = messages1["sc-and"]
        otherBShareList = messages0["invConv"]
        player.compare2(pq_vals0,pq_vals1,otherBShareList)
    ### 2. Comparison phase >>>>> ###

    ### <<<<< 3. Path evaluation phase ###
    player.resetMsgPoolWithKeys("shuffleReveal","optShuffle")#Reset message pool
    player.pathEval0_shuffleReveal(bShares,piShares)
    player.pathEval0_optShuffle(piShares)
    await player.distributeNetworkPool()#Send out messages
    print("4th Round-Evaluation: sending message.")

    if _id == 1:
        other0Data = await pool.recv("server0" )
        other1Data = await pool.recv("server2" )
        sigma  =  other0Data["shuffleReveal"]
        # print("Eval 1 msg: ", other1Data)
        gamma  =  other1Data["shuffleReveal"]
        gamma1 = other1Data["optShuffle"]
        player.resetMsgPoolWithKeys("shuffleReveal","optShuffle")
        player.pathEval1_shuffleRevealRespond(sigma,gamma,piShares)
        player.pathEval1_optShuffleRespond(gamma1,None,piShares)
    elif _id == 2:#(P0 to P2:sigma in optshuffle)
        message = await pool.recv("server0" )
        sigma  =  message["optShuffle"]
        player.resetMsgPoolWithKeys("shuffleReveal","optShuffle")
        player.pathEval1_optShuffleRespond(None,sigma,piShares)
    elif _id == 0:
        player.resetMsgPoolWithKeys("shuffleReveal","optShuffle")
        player.pathEval1_optShuffleRespond(None,None,piShares)
    await player.distributeNetworkPool()
    print("5th Round-Evaluation: bouncing back message")

    #ShuffleReveal: receive final reveal value 1->(0,2)
    #Optshuffle:    receive re-randomized RSS
    if _id == 0:
        otherData = await pool.recv("server1" )
        revealVec =  otherData["shuffleReveal"]
        otherShuffleShare = otherData["optShuffle"]
        # player.pathEval2(revealVec,otherShuffleShare)
        v00,v01 = player.pathEval2(revealVec,otherShuffleShare)
    elif _id == 1:
        message = await pool.recv("server2" )
        otherShuffleShare = message["optShuffle"]
        # player.pathEval2(None,otherShuffleShare)
        v10,v11 = player.pathEval2(None,otherShuffleShare)
        if BENCHMARK_CHECK_PROTOCOL_CORRECTNESS:
            await pool.send("server0",v11)
    elif _id == 2:
        other1Data = await pool.recv("server1" )
        revealVec =  other1Data["shuffleReveal"]
        other0Data = await pool.recv("server0" )
        otherShuffleShare = other0Data["optShuffle"]
        player.pathEval2(revealVec,otherShuffleShare)
    ### 3. Path evaluation phase >>>>> ###


    print("5th Round-Evaluation: completed")
    if BENCHMARK_MEASURE_ONLINE_COMMU:
        nowRecv = pool.getRecvBytes()
        print(nowRecv-lastRecv)
        lastRecv = nowRecv

    ### Needs to say, by this point the protocol is completed.
    ### But to reconstruct the final value we ask p1 send his v1 to p0 for a construction of final classification result.
    if BENCHMARK_CHECK_PROTOCOL_CORRECTNESS:
        if _id==0:
            v11 = await pool.recv("server1")
            print("The final selected class is: ", inRing(v00+ v01+v11,VEC_VAL_MAX_BOUND) )
    await pool.shutdown()

if __name__ == "__main__":
    _id = int(sys.argv[1])
    # _id = 0
    loop = asyncio.get_event_loop()
    loop.run_until_complete(async_main(_id))