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

The concrete idea may be seen in (currently inefficient) implementation.

The general idea behind LDL is as follows. We get embedding of size X, where X is power of 2,
and we will also return embedding of size X.
If input embedding is of size 1, we return it multiplied by learnable weight and adding learnable bias.
If it's bigger than 1: we'll called it _in_, and slice it in two equal parts with size X/2, called _in1_ and _in2_. We will now generate an embedding of size X/2, called _y1_ with 2 learnable weight vectors of size X/2, called _w11_ and _w12_, by _y1 := in1*w11 + in2*w12_. We also generate embedding _y2_ with learnable weight vectors _w21_ and _w22_ by _y2 := in1*w21 + in2+w22_.
The above makes sure that both _y1_ and _y2_ are based on all dimensions of _in_!
Now we can recursively call LDL on _y1_ and _y2_ generating output embeddings _out1_ and _out2_, with sizes X/2. We concatenate them into _out_, sized X, and return it.

The interpretation is: based on _in_ we split it into _in1_ and _in2_, mix them into "modules" _y1_ and _y2_, and compute _out1_ and _out2_ in those modules (and modules don't interact with each other computation-wise, in this layer). We might call it ModularDenseLayer, really.

This way, any output dimension may be any linear combination of input features (but not every output dimension may be any linear combination of input features, as they put constrainst on each other). We have log(X) depth of recursion, and on every recursion layer we have 2 * X learnable parameters, and 2 * X multiplications, so we have a complexity of O(X log X) where X is the size of layer/embedding.

It is worth noting that using LDL we can basically have way wider networks which compute more, altho they require more layers (as computation power of each layer is severly limited). The impact will be the biggest with bigger layers, maybe more than 10,000 neurons.

For comparison, a dense layer of size 1024->1024 have around 1M parameters. Using slightly less parameters per layer we could have a LDL layer 65536->65536, 64 times as wide!

With a dense layer of size 65536->65536, we have 4 billion parameters (currently unfeasible). But with less than 4 billiom parameters, with LDL we can process 2^27->2^27, that is 134217728->134217728, that is 130 million dimensions wide embedding.
