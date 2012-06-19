# plots factor graphs

import sys
import jsonpickle
import xmlrpclib

for a in sys.argv:
  print a

dn = jsonpickle.decode(sys.argv[1].replace("'",'"'))
fn = jsonpickle.decode(sys.argv[2].replace("'",'"'))
edges = jsonpickle.decode(sys.argv[3].replace("'",'"'))


print "plotting model with dimension nodes"
print dn
print "factor nodes"
print fn
print "factor node dimensions"
print edges



# Create an object to represent our server.
server_url = 'http://127.0.0.1:20738/RPC2'
server = xmlrpclib.Server(server_url)
G = server.ubigraph

G.clear()

node_list=[]
for ind, label in enumerate(dn):
  v = G.new_vertex()
  G.set_vertex_attribute(v, "label", label  )
  G.set_vertex_attribute(v, "fontsize", "15")
  G.set_vertex_attribute(v, "color", "#00FF00")
  node_list.append(v)

for label in fn:
  v = G.new_vertex()
  G.set_vertex_attribute(v, "label", label  )
  G.set_vertex_attribute(v, "fontsize", "15")
  G.set_vertex_attribute(v, "shape", "sphere")
  G.set_vertex_attribute(v, "color", "#ff0000")
  node_list.append(v)
  

for dn_ind, edge in enumerate(edges):
  for f in edge:
    e = G.new_edge( node_list[dn_ind], node_list[len(dn)+fn.index(f)] )

# for node_ind, parent_inds in enumerate(p):
#   if parent_inds != []:
#     for k in parent_inds:
#       e = G.new_edge(node_list[k-1], node_list[node_ind])
#       edge_lbl=set(l[k-1])-set(l[node_ind])
#       G.set_edge_attribute(e, "arrow", "true")
#       G.set_edge_attribute(e, "arrow_position", "0")
#       G.set_edge_attribute(e, "label", ''.join(edge_lbl))
