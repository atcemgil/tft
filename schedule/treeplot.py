import sys
import jsonpickle
import xmlrpclib

# Create an object to represent our server.
server_url = 'http://127.0.0.1:20738/RPC2'
server = xmlrpclib.Server(server_url)
G = server.ubigraph

G.clear()

p = jsonpickle.decode(sys.argv[1])
l = jsonpickle.decode(sys.argv[2].replace("'",'"'))
c = jsonpickle.decode(sys.argv[3])
global_min_path = jsonpickle.decode(sys.argv[4])

print "plotting model with p"
print p
print "labels"
print l

node_list=[]
for ind, label in enumerate(l):
  v = G.new_vertex()
  G.set_vertex_attribute(v, "label", label + " (" +str(c[ind])+ ")" )
  G.set_vertex_attribute(v, "fontsize", "15")
  if ind == 0:
    G.set_vertex_attribute(v, "color", "#ff0000")
  if (ind+1) in global_min_path:
    G.set_vertex_attribute(v, "color", "#00FF00")
  node_list.append(v)

for node_ind, parent_inds in enumerate(p):
  if parent_inds != []:
    for k in parent_inds:
      e = G.new_edge(node_list[k-1], node_list[node_ind])
      edge_lbl=set(l[k-1])-set(l[node_ind])
      G.set_edge_attribute(e, "arrow", "true")
      G.set_edge_attribute(e, "arrow_position", "0")
      G.set_edge_attribute(e, "label", ''.join(edge_lbl))


