#!/usr/bin/python3

from mininet.node import Controller
from mininet.net import Mininet
from mininet.cli import CLI
from mininet.link import TCLink
import time

def topology():
    net = Mininet(controller=Controller, link=TCLink)

    print("Creating nodes.")
    node1 = net.addHost('node1', ip='10.0.0.1')
    node2 = net.addHost('node2', ip='10.0.0.2')
    node3 = net.addHost('node3', ip='10.0.0.3')

    print("Creating switch.")
    switch = net.addSwitch('s1')

    print("Creating links.")
    net.addLink(node1, switch, bw=1, delay='10ms')
    net.addLink(node2, switch, bw=1, delay='10ms')
    net.addLink(node3, switch, bw=1, delay='10ms')

    print("Starting network.")
    net.start()

    output = node1.cmd('ping -v -c 3 10.0.0.3')
    print(output)

    print("node1 routing table")
    print(node1.cmd('route'))


    # node1.cmd('python3 online.py 0 2> online-player0.txt &')
    # node2.cmd('python3 online.py 1 2> online-player1.txt &')
    # node3.cmd('python3 online.py 2 2> online-player2.txt &')

    # output1 = node1.cmd('python3 online.py 0')
    # output2 = node2.cmd('python3 online.py 1')
    # output3 = node3.cmd('python3 online.py 2')
    # print(output1)
    # print(output2)
    # print(output3)

    # time.sleep(60)  # wait for 60 seconds



    # CLI(net)
    net.stop()

if __name__ == '__main__':
    topology()
