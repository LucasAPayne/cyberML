import numpy as np
from copy import deepcopy

class batch_provider:
    def __init__(self, data, batch_size, num_neg_samples = 2, seed = 1231245):
        '''
        Helper class to provide data in batches with negative examples.

        data: Training data triples
        batch_size: size of the mini-batches
        num_neg_samples: number of neg. samples.
        seed: random seed for neg. sample generation
        '''
        self.data = deepcopy(data)
        
        # The number of entities is the max of the 1st and 3rd column of data (subjects and objects)
        # The data is all IDs, from 0 to the number of subjects/objects, so finding the overall max of these gives the number of entities
        self.num_entities = np.max([np.max(data[:,0]), np.max(data[:,2])])

        np.random.seed(seed)
        np.random.shuffle(self.data)

        self.batch_size = batch_size
        self.number_minibatches = int(len(self.data)/batch_size)
        self.current_minibatch = 0

        self.num_neg_samples = num_neg_samples

    def next_batch(self):
        '''
        Return the next mini-batch.
        Data triples are shuffled after each epoch.
        '''
        i = self.current_minibatch
        di = self.batch_size
        
        # This minibatch is the piece of data starting from (current_minibatch_index * batch_size) until (next_minibatch_index * batch_size)
        mbatch = deepcopy(self.data[i*di:(i+1)*di])
        self.current_minibatch += 1
        
        if self.current_minibatch == self.number_minibatches:
            np.random.shuffle(self.data)
            self.current_minibatch = 0
        if self.num_neg_samples > 0:
            sub, rel, obj, labels = self.apply_neg_examples(list(mbatch[:,0]), list(mbatch[:,1]), list(mbatch[:,2]))
            return sub, rel, obj, labels
        else:
            # Return first 3 columns of batch data (subject, relation, object)
            # If batch_size is 100, this will return 100 triples
            return mbatch[:,0], mbatch[:,1], mbatch[:,2]

    def apply_neg_examples(self, sub, rel, obj):
        '''
        Generate neg. samples for a mini-batch.
        Both subject and object neg. samples are generated.
        '''
        vsize = len(sub)
        labels = np.array([1 for i in range(vsize)] + [-1 for i in range(self.num_neg_samples*2*vsize)])
        neg_sub = list(np.random.randint(self.num_entities, size = self.num_neg_samples*vsize))
        neg_obj = list(np.random.randint(self.num_entities, size = self.num_neg_samples*vsize))
        return np.concatenate([sub, neg_sub, sub*self.num_neg_samples]), np.concatenate([rel*(2*self.num_neg_samples+1)]), np.concatenate([obj, obj*self.num_neg_samples, neg_obj]), labels
