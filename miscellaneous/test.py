def dummyPercent(height,nodesNum):
    total = 2**height -1
    ret = (total-nodesNum)*100/total
    print(f'{ret:.2f}\%')

dummyPercent(5,11)

dummyPercent(6,19)

dummyPercent(7,21)

dummyPercent(10,138)

dummyPercent(12,159)

dummyPercent(15,167)

dummyPercent(18,369)