from sumolib import net
import os, sys
sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))

net_file = "C:/Users/DELL LATITUDE E7250/Desktop/Masters Research/v7/CenturyCityNode1.net.xml"
network = net.readNet(net_file)

for tls_id in network.getTLSIDs():
    print(f"\nTraffic Light: {tls_id}")
    connections = network.getTLS(tls_id).getConnections()
    outgoing_edges = set()
    for conn in connections:
        to_edge = conn.getToLane().getEdge()
        outgoing_edges.add(to_edge.getID())
    print("Outgoing edges:", list(outgoing_edges))
