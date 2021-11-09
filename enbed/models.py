import torch
import numpy as np
import matplotlib.pyplot as plt
from enbed.utils.scorer import RESCAL_score, DistMult_score

class RESCAL:
    def __init__(self, num_entities, num_relations, dim, seed = 1231245):
        '''
        Implementation of the RESCAL graph embedding model (Nickel et al., 2011).

        dim: embedding dimension
        num_entities: number of entities in the graph
        num_relations: number of relation types in the graph
        '''
        self.dim = dim
        self.num_entities = num_entities
        self.num_relations = num_relations

        # embeddings
        torch.manual_seed(seed)
        self.entities = torch.nn.Embedding(num_entities, dim)
        self.relations = torch.nn.Embedding(num_relations, dim*dim)
    
    def init(self):
        self.entities.weight.data *= 0.1
        self.relations.weight.data *= 0.1

    def score(self, sub, rel, obj):
        '''
        Score a list of triple [[s0, r0, o0], [s1, r1, o1],...]

        sub, rel and obj are lists [s0, s1, ...], [r0, r1, ...], [o0, o1, ...]
        '''
        s_emb = self.entities(torch.tensor(sub).long())
        o_emb = self.entities(torch.tensor(obj).long())
        r_emb = self.relations(torch.tensor(rel).long())

        return RESCAL_score(s_emb, o_emb, r_emb.view(-1, self.dim, self.dim))

    def prob(self, sub, rel, obj):
        '''
        Apply sigmoid to score.
        '''
        return torch.sigmoid(self.score(sub, rel, obj))

    def save(self, savepath, appdix = ''):
        '''
        Save and visualize embeddings.
        '''
        rel_embs = self.relations.weight.data.detach().numpy()
        ent_embs = self.entities.weight.data.detach().numpy()
        np.save('{}/relation_embeddings_{}.npy'.format(savepath, appdix), rel_embs)
        np.save('{}/entity_embeddings_{}.npy'.format(savepath, appdix), ent_embs)

        plt.close()
        for j in range(50):
            plt.vlines(ent_embs[j], j+0.1, (j+1)-0.1)
        plt.savefig('{}/entity_embeddings_{}.png'.format(savepath, appdix))

        plt.close()
        for j in range(len(rel_embs)):
            plt.vlines(rel_embs[j], j+0.1, (j+1)-0.1)
        plt.savefig('{}/relation_embeddings_{}.png'.format(savepath, appdix))


class Energy(RESCAL):
    def __init__(self, num_entities, num_relations, dim, seed = 1231245):
        '''
        Energy-based model for calculating embeddings. The cost is obtained using stochastic sampling.
        '''
        super().__init__(num_entities, num_relations, dim, seed)
        np.random.seed(seed)

    def cost(self, sub, rel, obj, num_samples, burnin=0):
        '''
        Cost function using sampling to maximize data likelihood.
        '''
        # nbatch is the dimension of embedding vectors
        nbatch = len(sub)
        pscore = self.score(sub, rel, obj)

        total_score = 0
        old_score = pscore
        for k in range(num_samples+burnin):
            sro = np.random.randint(3, size=nbatch)
            # One mask will be 1 and the rest will be 0 (randomly) on each for loop
            smask = (sro == 0)
            rmask = (sro == 1)
            omask = (sro == 2)
            
            # Note: '~' is bitwise negation; 0 -> -1, 1 -> -2
            # Pick new sub, obj, rel by first multiplying by the complement of the mask, then adding a random offset to one of them
            new_sub = ~smask*sub + smask*(np.random.random(nbatch)*self.num_entities)
            new_obj = ~omask*obj + omask*(np.random.random(nbatch)*self.num_entities)
            new_rel = ~rmask*rel + rmask*(np.random.random(nbatch)*self.num_relations)

            # Score the new triple
            proposal_score = self.score(new_sub, new_rel, new_obj)

            # filters is a binary tensor containing the result of comparing each element of a random tensor and the tensor resulting from
            # Ti = e^(proposal_score_i - old_score_i) for each element i
            # Recalculate old_score by applying filters to proposal_score and old_score and combining the results
            filters = 1.*(torch.rand(nbatch) <= torch.exp(proposal_score-old_score))
            old_score = proposal_score*filters + old_score*(1-filters)

            # Convert filters to a numpy array and construct a new triple by applying filters to the old and new triples and combining the results
            filters = filters.detach().numpy()
            sub = np.array(new_sub*filters + sub*(1-filters), dtype=int)
            obj = np.array(new_obj*filters + obj*(1-filters), dtype=int)
            rel = np.array(new_rel*filters + rel*(1-filters), dtype=int)

            # Convert old_score from tensor to scalar by summing its elements
            # Add the old score to the sum, not counting burnin
            if k >= burnin:
                total_score += old_score.sum()

        # The cost is the negative sum of the original score tensor plus a small offset based on the number of samples and the total score
        # Cost is a scalar. Higher total score gives lower cost.
        cost = -pscore.sum() + 1./num_samples*total_score
        return cost

class EnergyDiag(Energy):
    def __init__(self, num_entities, num_relations, dim, seed = 1231245):
        '''
        Energy-based model for calculating embeddings, with diagonally-constrained relation matrices.
        Similar to DistMult (Yang et al., 2014).
        '''
        super().__init__(num_entities, num_relations, dim, seed)
        self.relations = torch.nn.Embedding(num_relations, dim)

    def score(self, sub, rel, obj):
        s_emb = self.entities(torch.tensor(sub).long())
        o_emb = self.entities(torch.tensor(obj).long())
        r_emb = self.relations(torch.tensor(rel).long())

        return DistMult_score(s_emb, o_emb, r_emb)
