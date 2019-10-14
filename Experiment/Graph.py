import networkx as nx

def create_graph_CNN(skip_hyperconnection_config = None):
    G = nx.DiGraph()
    # Vanilla
    G.add_node('IoT')
    G.add_node('e')
    G.add_node('f')
    G.add_node('c')
    
    G.add_edge('IoT','e')
    G.add_edge('e','f')
    G.add_edge('f','c')
    
    if skip_hyperconnection_config: # deepFogGuard and ResiliNet
        if skip_hyperconnection_config[0] == 1:
            G.add_edge('IoT','f')
        if skip_hyperconnection_config[1] == 1:
            G.add_edge('e','c')
    return G

def create_graph_MLP_health(skip_hyperconnection_config = None):
    G = nx.DiGraph()
    # Vanilla
    G.add_node('IoT')
    G.add_node('e')
    G.add_node('f1')
    G.add_node('f2')
    G.add_node('c')
    
    G.add_edge('IoT','e')
    G.add_edge('e','f2')
    G.add_edge('f2','f1')
    G.add_edge('f1','c')

    if skip_hyperconnection_config: # deepFogGuard and ResiliNet
        if skip_hyperconnection_config[0] == 1:
            G.add_edge('IoT','f2')
        if skip_hyperconnection_config[1] == 1:
            G.add_edge('e','f1')
        if skip_hyperconnection_config[2] == 1:
            G.add_edge('f2','c')
    return G

def create_graph_MLP_camera(skip_hyperconnection_config = None):
    G = nx.DiGraph()
    # Vanilla
    G.add_node('cams') # dummy node to connect to all edge nodes
    G.add_node('e1')
    G.add_node('e2')
    G.add_node('e3')
    G.add_node('e4')
    G.add_node('f1')
    G.add_node('f1')
    G.add_node('f3')
    G.add_node('f4')
    G.add_node('c')

    G.add_edge('cams','e1')
    G.add_edge('cams','e2')
    G.add_edge('cams','e3')
    G.add_edge('cams','e4')
    G.add_edge('e1','f3')
    G.add_edge('e2','f4')
    G.add_edge('e3','f4')
    G.add_edge('e4','f4')
    G.add_edge('f3','f2')
    G.add_edge('f4','f2')
    G.add_edge('f2','f1')
    G.add_edge('f1','c')
    
    if skip_hyperconnection_config: # deepFogGuard and ResiliNet
        if skip_hyperconnection_config[0] == 1:
            G.add_edge('e4','f2')
        if skip_hyperconnection_config[1] == 1:
            G.add_edge('e3','f2')
        if skip_hyperconnection_config[2] == 1:
            G.add_edge('e2','f2')
        if skip_hyperconnection_config[3] == 1:
            G.add_edge('e1','f2')
        if skip_hyperconnection_config[4] == 1:
            G.add_edge('f4','f1')
        if skip_hyperconnection_config[5] == 1:
            G.add_edge('f3','f1')
        if skip_hyperconnection_config[6] == 1:
            G.add_edge('f2','c')
    return G

def fail_node_graph(graph, node_failure_combination, exp):
    if exp == "CIFAR/Imagenet":
        nodes = ["f", "e"]
    if exp == "Health":
        nodes = ["f1", "f2", "e"]
    if exp == "Camera":
        nodes = ["f1", "f2", "f3", "f4", "e1", "e2", "e3", "e4"]
    for index, node in enumerate(node_failure_combination):
        if node == 0: # if dead
            graph.remove_node(nodes[index])

def identify_no_information_flow_graph(graph, exp):
    if exp == "CIFAR/Imagenet":
        return not nx.has_path(graph, 'IoT', 'c')
    if exp == "Health":
        return not nx.has_path(graph, 'IoT', 'c')
    if exp == "Camera":
        return not nx.has_path(graph, 'cams', 'c')