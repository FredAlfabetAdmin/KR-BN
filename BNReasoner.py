from typing import Union
from BayesNet import BayesNet
import pandas as pd

# I guess dat dit werkt, alleen weet ik niet helemaal hoe ik met bNRReasoner moet werken dus de structuur klopt niet helemaal...
class BNReasoner:
    def __init__(self, net: Union[str, BayesNet]):
        """
        :param net: either file path of the bayesian network in BIFXML format or BayesNet object
        """
        if type(net) == str:
            # constructs a BN object
            self.bn = BayesNet()
            # Loads the BN from an BIFXML file
            self.bn.load_from_bifxml(net)
        else:
            self.bn = net

    # TODO: This is where your methods should go
    # print("hi")

def run(Queri, evidence):
    bn = BayesNet()
    bn.load_from_bifxml("testing/lecture_example.BIFXML")
    # BayesNet.draw_structure(bn)
   #Is needed for pd.Series
    e = []
    for k in evidence:
        e.append(k)
    Network_Pruning(bn, Queri, pd.Series(data= evidence, index = e))
    # BayesNet.draw_structure(bn)
    # print(BayesNet.get_all_cpts(bn))
    # print(BayesNet.get_all_variables(bn))

def Network_Pruning(bn, Q, evidence):

    #Edge Pruning
    children = dict()
    e = list()

    #Get edges between nodes by asking for children, which are save in a list (children)
    #Adds the evidence to list e such that later on, these nodes will not be removed from the BN
    for u, v in evidence.items():
        children[u] = (BayesNet.get_children(bn, u))
        e.append(u)    

    #Remove edges between evidence and children
    #Replaces the factors/cpt to the reduced factors/cpt 
    for key in children:
        for value in children[key]:
            BayesNet.del_edge(bn,(key,value))
            BayesNet.update_cpt(bn, value, BayesNet.reduce_factor(evidence, BayesNet.get_cpt(bn, value)))
   
    #Node Pruning
    #Need to keep removing leafnodes untill all leafnodes that can be removed are removed
    i = 1
    while i > 0:
        i = 0
        var = BayesNet.get_all_variables (bn)
        for v in var:
            child = BayesNet.get_children(bn, v)
            #If node is a leaf node and not in the Q or e, remove from bn
            if len(child) == 0 and v not in Q and v not in e:
                BayesNet.del_var(bn, v)                
                i += 1   
    
if __name__ == "__main__":
    run(["Rain?"],{"Winter?": True})

    
