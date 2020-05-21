import networkx as nx
import matplotlib.pyplot as plt
g = nx.Graph()
g.add_node("Obama")
g.add_node("Trump")

g.add_nodes_from(["Hillary","Michelle"])


g.nodes()
g.add_edge("Obama","Michelle")
g.add_edge("Hillary","Clinton")
g.add_edge("Trump","Ivanka")
g.add_edge("Trump","Clinton")
g.add_edge("Obama","Mitt")

g.add_edges_from([("Hillary","Clinton"),("Obama","Trump"),("Obama","Clinton"),("Michelle","Mitt")])


g.nodes()
g.edges()


nx.info(g)
nx.draw(g)

nx.draw(g,with_labels=True)
nx.draw_networkx(g)

nx.spring_layout(g)

nx.degree(g,"Trump")
nx.degree(g,"Obama")
nx.degree(g,"Clinton")

print(nx.degree_centrality(g))
nx.eigenvector_centrality(g)
print(nx.eigenvector_centrality(g))

nx.betweenness_centrality(g)
print(nx.betweenness_centrality(g))

plt.show()