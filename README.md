# sparser-layer

This is a proof-of-concept, naive implementation of LessDenseLayer, i.e. a modified Dense Layer which is less dense.

# Motivation

I always think of dense layer as a function of embedding to embedding.
The number of dimensions (neurons) is, in good approximation, linearly proportional to information in this embedding.
This means, if we want to represent twice as much information, we have to double the size of the layer.
The problem with this is, if we want to double the information in every layer,
we have to quadruple the number of parameters and computation time.
So, for N bits of information we have O(N²) time and space complexity, and this is IMO terrible.

I think this is why the practical limit on number of neurons is around 1,000 - 10,000 per layer.

I propose better Dense Layer, LessDenseLayer, which for N neurons has O(NlogN) parameters, space & time complexity,
while keeping the most important features of DenseLayer. This is (at the time of writing) basically proof-of-work, untested.

The most important features of Dense Layer function, which we will keep with LDL, are:
* It has to have learnable parameters.
* It has to be continuous and differentiable.
* It should be symmetrical - every neuron is treated the same way.  (altho, maybe, it doesn't have to?)
* Every output dimension is (or: may be) influenced by every input dimension.

I'll describe the method in the next commit.
