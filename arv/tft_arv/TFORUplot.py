# plots factor graphs

import sys
import jsonpickle
import xmlrpclib
import time

for a in sys.argv:
  print a

f = open(sys.argv[1], 'r')

vertices = jsonpickle.decode(f.readline())
edges = jsonpickle.decode(f.readline())


print "plotting model with vertices"
print vertices
print "edges"
print edges


# Create an object to represent our server.
server_url = 'http://127.0.0.1:20738/RPC2'
server = xmlrpclib.Server(server_url)
G = server.ubigraph

G.clear()

node_list=[]
for ind, label in enumerate(vertices):
    if ind % 1000 == 0:
        print "vertex index:", ind
    v = G.new_vertex()
    if not '_data' in label:
        G.set_vertex_attribute(v, "label", label  )
    G.set_vertex_attribute(v, "fontsize", "10")
    G.set_vertex_attribute(v, "color", "#00FF00")
    node_list.append(v)
    #print 'append',label

print len(node_list)
print len(edges)
for i in range(0,len(edges),2):
    e = G.new_edge( node_list[edges[i]-1], node_list[edges[i+1]-1] )
