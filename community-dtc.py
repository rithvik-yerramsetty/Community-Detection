import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import os
import seaborn as sb
import networkx as nx
from copy import deepcopy
import pandas as pd
import time


if not os.path.exists('../plots'):
    os.makedirs('../plots')

question = 1
q_dict = {1:'fb', 2:'btc'}

def plot_one_iter(fielder_vec, nodes_connectivity_list, adj_mat, graph_partition):
    os.environ['QT_QPA_PLATFORM'] = 'xcb'
    sorted_fiedler = np.sort(fielder_vec, axis=0).reshape(-1)
    sort_nodes = np.argsort(fielder_vec, axis=0).reshape(-1)
    
    plt.figure()
    plt.scatter(x=np.arange(len(np.unique(nodes_connectivity_list))), y=sorted_fiedler)
    plt.savefig('../plots/' + q_dict[question] + '_fielder_one_iter_spectral.png')

    adj_mat_sorted = deepcopy(adj_mat)
    adj_mat_sorted = np.take(adj_mat_sorted, sort_nodes, axis =  1)
    adj_mat_sorted = np.take(adj_mat_sorted, sort_nodes, axis =  0)

    # plt.figure()
    # plt.imsave('../plots/' + q_dict[question] + '_adj_mat_one_iter_spectral.png', adj_mat_sorted, cmap = 'plasma')
    
    plt.figure()
    plt.spy( adj_mat_sorted)
    plt.savefig('../plots/' + q_dict[question] + '_adj_mat_one_iter_spectral.png')
    # ax = sb.heatmap(adj_mat_sorted)
    # plt.savefig('../plots/adj_mat_one_iter_'+ q_dict[question] + '.png')

    plt.figure()
    G = nx.Graph(adj_mat)
    pos = nx.spring_layout(G)
    nx.draw(G=G, pos = pos, node_color = graph_partition[:,1], cmap=plt.cm.Set1, with_labels= False)
    plt.savefig('../plots/' + q_dict[question] + '_graph_one_iter_spectral.png')

    del os.environ['QT_QPA_PLATFORM']
    return

def plot_louvain(graph_partition, nodes_connectivity_list):

    index_sorted = np.argsort(graph_partition[:,1].reshape(-1))
    node_ids_sorted = graph_partition[:,0][index_sorted]
   
    num_nodes = len(np.unique(nodes_connectivity_list))
    node_vec = np.unique(nodes_connectivity_list).reshape(-1,1)
    
    node_vec_dict = {value: index for index, value in enumerate(node_vec.reshape(-1))}

    first_nodes = [node_vec_dict[value] for value in nodes_connectivity_list[:,0]]
    second_nodes = [node_vec_dict[value] for value in nodes_connectivity_list[:,1]]

    adj_mat = np.zeros((num_nodes,num_nodes),dtype=int)
    adj_mat[first_nodes, second_nodes] = 1
    adj_mat[second_nodes, first_nodes] = 1


    adj_mat_sorted = deepcopy(adj_mat)
    adj_mat_sorted = np.take(adj_mat_sorted, node_ids_sorted, axis =  1)
    adj_mat_sorted = np.take(adj_mat_sorted, node_ids_sorted, axis =  0)

    os.environ['QT_QPA_PLATFORM'] = 'xcb'

    plt.figure()
    plt.spy( adj_mat_sorted)
    plt.savefig('../plots/' + q_dict[question] + '_adj_mat_sorted_louvian.png')

    plt.figure()
    G = nx.Graph(adj_mat_sorted)
    pos = nx.spring_layout(G)
    nx.draw(G=G, pos = pos, node_color = graph_partition[:,1], cmap=plt.cm.Set1, with_labels= False)
    plt.savefig('../plots/' + q_dict[question] + '_graph_louvian.png')
    del os.environ['QT_QPA_PLATFORM']

def import_facebook_data(path):
    data =  np.unique(np.sort(np.genfromtxt(path, dtype=int), axis=1), axis=0)
    return data

def spectralDecomp_OneIter(nodes_connectivity_list):
    num_nodes = len(np.unique(nodes_connectivity_list))
    # print('num nodes- ', num_nodes)
    node_vec = np.unique(nodes_connectivity_list).reshape(-1,1)
    
    node_vec_dict = {value: index for index, value in enumerate(node_vec.reshape(-1))}

    first_nodes = [node_vec_dict[value] for value in nodes_connectivity_list[:,0]]
    second_nodes = [node_vec_dict[value] for value in nodes_connectivity_list[:,1]]

    adj_mat = np.zeros((num_nodes,num_nodes), dtype=int)
    adj_mat[first_nodes, second_nodes] = 1
    adj_mat[second_nodes, first_nodes] = 1

    deg_mat = np.diag(np.sum(adj_mat, axis=1))
    lap_mat = deg_mat - adj_mat

    
    eigen_values, eigen_vectors = eigh(a=lap_mat, b=deg_mat)
    sorted_eig_ind = np.argsort(eigen_values.reshape(-1))
    fiedler_vec = eigen_vectors[:,sorted_eig_ind][:,1]


    par_vec = np.empty((num_nodes,1), dtype=int)
    if(len(np.unique(fiedler_vec.reshape(-1) <= 0)) < 2):
        return [], [], []
    

    community_values = np.unique(fiedler_vec <= 0, return_index=True)
    community_id1 = node_vec[community_values[1][community_values[0] == True][0]]
    community_id2 = node_vec[community_values[1][community_values[0] == False][0]]
    par_vec[fiedler_vec <= 0] , par_vec[fiedler_vec > 0] = community_id1, community_id2

    
    graph_part = np.concatenate((node_vec, par_vec), axis = 1)

    return fiedler_vec, adj_mat, graph_part

def splitNodeConnectivityList(nodes_connectivity_list, graph_part):
    if(len(graph_part) == 0 ):
        return [],[]
    communities = np.unique(graph_part[:,1])
    if(len(communities) < 2):
        return [],[]

    part_one = graph_part[:,0][graph_part[:,1] == communities[0]]
    part_two = graph_part[:,0][graph_part[:,1] == communities[1]]

    
    nodes_comm_one = np.array([1 if (edge[0] in part_one and edge[1] in part_one) else 0 for edge in nodes_connectivity_list])
    nodes_comm_two = np.array([1 if (edge[0] in part_two and edge[1] in part_two) else 0 for edge in nodes_connectivity_list])

    node_list_one = np.empty((np.sum(nodes_comm_one), 2), dtype=int)
    node_list_two = np.empty((np.sum(nodes_comm_two), 2), dtype=int)


    node_list_one = nodes_connectivity_list[nodes_comm_one > 0]
    node_list_two = nodes_connectivity_list[nodes_comm_two > 0]


    return node_list_one, node_list_two

def isStable(fiedler_vec):
    k = 200

    sorted_fiedler = np.sort(fiedler_vec)
    # diff_fiedler = np.diff(sorted_fiedler)

    t = []
    for i in range(len(sorted_fiedler)-1):
        t.append(sorted_fiedler[i+1] - sorted_fiedler[i])

    diff_fiedler = np.array(t)

    max_diff = np.max(diff_fiedler)
    

    mean_ = np.mean(diff_fiedler)
    
    
    if(max_diff < k*mean_):
        return False
    
    return True

def spectralDecomp(nodes_connectivity_list):
    fielder_vec, adj_mat, graph_part = spectralDecomp_OneIter(nodes_connectivity_list)
    nodes_connectivity_list_1, nodes_connectivity_list_2 =  splitNodeConnectivityList(nodes_connectivity_list, graph_part)

    if(len(nodes_connectivity_list_1) != 0  and len(nodes_connectivity_list_2) != 0 and isStable(fielder_vec)):    
        graph_part_1 = spectralDecomp(nodes_connectivity_list_1)
        graph_part_2 = spectralDecomp(nodes_connectivity_list_2)
        temp_graph_part = graph_part
        if(len(graph_part_1) != 0  and len(graph_part_2) != 0):
            graph_part = np.row_stack((graph_part_1, graph_part_2))

            nodes_total = np.unique(nodes_connectivity_list).reshape(-1)
            nodes_1 = np.unique(nodes_connectivity_list_1).reshape(-1,1)
            nodes_2 = np.unique(nodes_connectivity_list_2).reshape(-1,1)
            nodes_after = np.row_stack((nodes_1, nodes_2)).reshape(-1)

            for node in nodes_total:
                if(node not in nodes_after):
                    idx = np.where(temp_graph_part[:,0] == node)[0]
                    graph_part = np.row_stack((graph_part, temp_graph_part[idx]))

    return graph_part

def spectralDecomposition(nodes_connectivity_list):
    graph_part = spectralDecomp(nodes_connectivity_list)

    print('Total communities - ' ,len(np.unique(graph_part[:,1])))
    print('Communities:')
    print(np.unique(graph_part[:,1]))

    return graph_part

def createSortedAdjMat(graph_partition, nodes_connectivity_list):

    index_sorted = np.argsort(graph_partition[:,1].reshape(-1))
    node_ids_sorted = graph_partition[:,0][index_sorted]
   
    num_nodes = len(np.unique(nodes_connectivity_list))
    node_vec = np.unique(nodes_connectivity_list).reshape(-1,1)
    
    node_vec_dict = {value: index for index, value in enumerate(node_vec.reshape(-1))}

    first_nodes = [node_vec_dict[value] for value in nodes_connectivity_list[:,0]]
    second_nodes = [node_vec_dict[value] for value in nodes_connectivity_list[:,1]]

    adj_mat = np.zeros((num_nodes,num_nodes),dtype=int)
    adj_mat[first_nodes, second_nodes] = 1
    adj_mat[second_nodes, first_nodes] = 1


    adj_mat_sorted = deepcopy(adj_mat)
    adj_mat_sorted = np.take(adj_mat_sorted, node_ids_sorted, axis =  1)
    adj_mat_sorted = np.take(adj_mat_sorted, node_ids_sorted, axis =  0)

    os.environ['QT_QPA_PLATFORM'] = 'xcb'

    # plt.figure()
    # plt.imsave('../plots/' + q_dict[question] + '_adj_mat_sorted_spectral.png', adj_mat_sorted, cmap = 'plasma')
    # sb.heatmap(adj_mat_sorted)
    # plt.savefig('../plots/adj_mat_sorted_' + q_dict[question] + '.png')
    plt.figure()
    plt.spy( adj_mat_sorted)
    plt.savefig('../plots/' + q_dict[question] + '_adj_mat_sorted_spectral.png')

    plt.figure()
    G = nx.Graph(adj_mat_sorted)
    pos = nx.spring_layout(G)
    nx.draw(G=G, pos = pos, node_color = graph_partition[:,1], cmap=plt.cm.Set1, with_labels= False)
    plt.savefig('../plots/' + q_dict[question] + '_graph_spectral.png')

    del os.environ['QT_QPA_PLATFORM']

def louvain_one_iter(nodes_connectivity_list):
    nodes = np.unique(nodes_connectivity_list)
    num_nodes = len(nodes)
    M = 2*len(nodes_connectivity_list)
    adj_mat = np.zeros((num_nodes,num_nodes))
    adj_mat[nodes_connectivity_list[:,0], nodes_connectivity_list[:,1]] = 1/M
    adj_mat[nodes_connectivity_list[:,1], nodes_connectivity_list[:,0]] = 1/M

    deg_vec = np.sum(adj_mat, axis=1)

    communities = [[i] for i in nodes]
    dict = {i:i for i in nodes}
    # print(dict.values().shape)
    # print(len(set(dict.values())))
    i = 1
    changes=0

    while(True):
        i+=1
        flag = 0
        changes=0
        # print(len(set(dict.values())))
        for node in nodes:
            
            neighbour_nodes, _ = np.nonzero(adj_mat[node,:].reshape(-1,1))
            node_community = communities[dict[node]]

            
            sig_tot_node = sum(deg_vec[node_community])

            k_i = deg_vec[node]

            k_i_out = 2*np.sum(adj_mat[node][node_community])

            delta_Q_demerge = 2*k_i*sig_tot_node- 2*k_i**2 - k_i_out
            
            max_delta_Q = 0
            best_community = dict[node]

            neigh_communities = []

            for neigh in neighbour_nodes:
                neigh_communities.append(dict[neigh])
            
            neigh_communities = np.unique(np.array(neigh_communities, dtype = int))

            for neigh_com in neigh_communities:
                if(neigh_com == dict[node]):
                    continue

                neigh_community_nodes = communities[neigh_com]

                
                sig_tot_neigh = sum(deg_vec[neigh_community_nodes])
                
                k_i_in_merge = 2*np.sum(adj_mat[node][neigh_community_nodes])

                delta_Q_merge = k_i_in_merge - 2*sig_tot_neigh*k_i

                delta_Q = delta_Q_demerge + delta_Q_merge

                if(delta_Q > max_delta_Q):
                    max_delta_Q = delta_Q
                    best_community = neigh_com
            
            if(best_community != dict[node]):
                if node in communities[dict[node]]:
                    communities[dict[node]].remove(node)

                if node not in communities[best_community]:
                    communities[best_community].append(node)

                dict[node] = best_community

                flag = 1
                changes+=1
        if(flag == 0):
            break

    graph_part = np.array([list(dict.keys()), list(dict.values())], dtype=int).T

    print('Total communities - ' ,len(np.unique(graph_part[:,1])))
    print('Communities:')
    print(np.unique(graph_part[:,1]))

    return graph_part

def import_bitcoin_data(path):
    df = pd.read_csv(path, header=None)
    # df = df[df[2] > 0]
    df = df.drop(df.columns[[2, 3]], axis = 1)
    
    data = df.to_numpy()
    data = np.sort(data, axis = 1)
    data = np.unique(data, axis = 0)
    nodes = np.unique(data)
    dict = {value : ind  for ind, value in enumerate(nodes)}

    for  i in range(len(data)):
        data[i][0] = dict[data[i][0]]
        data[i][1] = dict[data[i][1]]

    # print(np.unique(data))
    return data

def get_modularity(adj_mat, graph_partition):
    G = nx.Graph(adj_mat)
    
    communitiy_ids = np.unique(graph_partition[:,1])
    communities = []


    for i in communitiy_ids:
        community_i = set(graph_partition[:,0][np.where(graph_partition[:,1] == i)[0]])
        communities.append(community_i)

    modularity = nx.community.modularity(G, communities)
    return modularity

if __name__ == "__main__":


    print("################ FACEBOOK DATA ####################")
    nodes_connectivity_list_fb = import_facebook_data("../data/facebook_combined.txt")

    
    fielder_vec_fb, adj_mat_fb, graph_partition_fb = spectralDecomp_OneIter(nodes_connectivity_list_fb)
    plot_one_iter(fielder_vec=fielder_vec_fb, nodes_connectivity_list=nodes_connectivity_list_fb, adj_mat=adj_mat_fb, graph_partition=graph_partition_fb)
    

    print('\nMethod: Spectral Decomposition')
    start_time = time.time()
    graph_partition_fb = spectralDecomposition(nodes_connectivity_list_fb)
    end_time = time.time()

    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time} seconds")
    modularity_spectral_fb = get_modularity(adj_mat=adj_mat_fb, graph_partition = graph_partition_fb)
    print("Modularity: ", modularity_spectral_fb)


    clustered_adj_mat_fb = createSortedAdjMat(graph_partition_fb, nodes_connectivity_list_fb)


    print('\nMethod: Louvain')
    start_time = time.time()
    graph_partition_louvain_fb = louvain_one_iter(nodes_connectivity_list_fb)
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
    modularity_louvian_fb = get_modularity(adj_mat=adj_mat_fb, graph_partition = graph_partition_louvain_fb)
    print("Modularity: ", modularity_louvian_fb)

    plot_louvain(graph_partition_louvain_fb, nodes_connectivity_list_fb)


    # ############ Answer qn 1-4 for bitcoin data #################################################
    # # Import soc-sign-bitcoinotc.csv
    print("################ BITCOIN DATA ####################")
    question+=1
    nodes_connectivity_list_btc = import_bitcoin_data("../data/soc-sign-bitcoinotc.csv")

    # # Question 1
    fielder_vec_btc, adj_mat_btc, graph_partition_btc = spectralDecomp_OneIter(nodes_connectivity_list_btc)
    plot_one_iter(fielder_vec=fielder_vec_btc, nodes_connectivity_list=nodes_connectivity_list_btc, adj_mat=adj_mat_btc, graph_partition=graph_partition_btc)

    # # Question 2
    print('\nMethod: Spectral Decomposition')
    start_time = time.time()
    graph_partition_btc = spectralDecomposition(nodes_connectivity_list_btc)
    end_time = time.time()

    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time} seconds")

    modularity_spectral_btc = get_modularity(adj_mat=adj_mat_btc, graph_partition = graph_partition_btc)
    print("Modularity: ", modularity_spectral_btc)

    # # Question 3
    clustered_adj_mat_btc = createSortedAdjMat(graph_partition_btc, nodes_connectivity_list_btc)

    # # Question 4
    print('\nMethod: Louvain')
    start_time = time.time()

    graph_partition_louvain_btc = louvain_one_iter(nodes_connectivity_list_btc)
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
    modularity_louvian_btc = get_modularity(adj_mat=adj_mat_btc, graph_partition = graph_partition_louvain_btc)
    print("Modularity: ", modularity_louvian_btc)
    plot_louvain(graph_partition_louvain_btc, nodes_connectivity_list_btc)



