# mat.py
# (C) Jeff Orchard, University of Waterloo, 2025

import numpy as np
import copy


'''
=========================================

 Mat class

=========================================
'''

class Mat(object):
    def __init__(self, val: np.ndarray):
        '''
         v = Mat(val)

         Creates a Mat object for the 2D numpy array val.

         Inputs:
           val   a 2D numpy array, dimensions DxN

         Output:
           v     object of Mat class

         Then, we can get its value using any of:
           v.val
           v()
         both of which return a numpy array.

         The member v.creator is either None, or is a reference
         to the MatOperation object that created v.

         The object also stores gradient information in v.grad.

         Usage:
           v = Mat(np.array([[1,2],[3,4.]]))
           len(v)  # returns number of rows
           v()     # returns v.val (a numpy array)
        '''
        self.val = np.array(copy.deepcopy(val), ndmin=2)
        self.rows, self.cols = np.shape(self.val)
        self.grad = np.zeros_like(self.val)
        self.creator = None

    def set(self, val: np.ndarray):
        self.val = np.array(val, dtype=float, ndmin=2)

    def zero_grad(self):
        self.grad = np.zeros_like(self.val)
        if self.creator!=None:
            self.creator.zero_grad()

    def backward(self, s: np.array =None):
        if s is None:
            s = np.ones_like(self.val)
        self.grad = self.grad + s
        if self.creator!=None:
            self.creator.backward(s)

    def __len__(self) -> int:
        return self.rows

    def __call__(self) -> np.ndarray:
        return self.val

    def __str__(self):
        return str(self.val)

    def __repr__(self):
        return f'Mat({self.val})'




'''
=========================================

 MatOperation

=========================================
'''
class MatOperation():
    '''
     op = MatOperation()

     MatOperation is an abstract base class for mathematical operations
     on matrices... Mat objects in particular.

     The MatOperation object op stores its arguments in the list op.args,
     and has the functions:
       op.__call__()
       op.zero_grad()
       op.backward()

     Usage:
       op()  # evaluates the operation without re-evaluating the args
       op.zero_grad() # resets grad to zero for all the args
       op.backward(s)  # propagates the derivative to the Vars below
    '''
    def __init__(self):
        self.args = []

    def __call__(self):
        raise NotImplementedError

    def zero_grad(self):
        for a in self.args:
            a.zero_grad()

    def backward(self, s=1.):
        raise NotImplementedError


'''
=========================================

 Operation Implementations

=========================================
'''

class Plus(MatOperation):
    '''
     Implements adding two Mat objects.
     Usage:
     > fcn = Plus()
     > C = fcn(A, B)  # performs C = A + B
    '''
    def __call__(self, a: Mat, b: Mat) -> Mat:
        assert a.rows==b.rows and a.cols==b.cols, 'The shape of the arguments to Plus do not match'
        self.args = [a, b]
        v = Mat(self.args[0].val + self.args[1].val)
        v.creator = self
        return v

    def backward(self, s=None):
        self.args[0].backward(s*np.ones_like(self.args[0].val))
        self.args[1].backward(s*np.ones_like(self.args[1].val))

class Minus(MatOperation):
    '''
     Implements subtracting two Mat objects.
     Usage:
     > fcn = Minus()
     > C = fcn(A, B)  # performs C = A - B
    '''
    def __call__(self, a: Mat, b: Mat) -> Mat:
        self.args = [a, b]
        v = Mat(self.args[0].val - self.args[1].val)
        v.creator = self
        return v

    def backward(self, s=None):
        self.args[0].backward(s*np.ones_like(self.args[0].val))
        self.args[1].backward(-s*np.ones_like(self.args[1].val))

##########################################################  

class ReLU(MatOperation):
    '''
     Implements the ReLU activation function for Mat objects.
     Usage:
     > fcn = ReLU()
     > C = fcn(A)  # A and C are Mat objects
    '''
    def __call__(self, x: Mat) -> Mat:
        self.args = [x]
        v = Mat(np.maximum(0, x.val))
        v.creator = self
        return v

    def backward(self, s=1.):
        grad = np.where(self.args[0].val > 0, 1, 0)
        self.args[0].backward(s * grad)



class Mul(MatOperation):
    '''
     af = Mul()

     Creates a MatOperation object that represents the matrix-matrix
     product.

     Usage:
      > af = Mul()
      > y = af(A, B)  # where A and B are both Mat objects
    '''
    def __call__(self, a: Mat, b: Mat) -> Mat:
        self.args = [a,b]
        v = Mat(self.args[0].val @ self.args[1].val)
        v.creator = self
        return v

    def backward(self, s: np.ndarray):
        grad_A = s @ self.args[1].val.transpose()
        grad_B = self.args[0].val.transpose() @ s
        self.args[0].backward(grad_A)
        self.args[1].backward(grad_B)

class MSE(MatOperation):
    '''
     mserr = MSE()

     Creates a mean squared error function, with the following specifications:
      loss = mserr(y, target)

     where
      y       Mat object with output vectors in rows
      target  Mat object with one-hot target vectors in rows

     Returns the mean squared error as a (1,1) Mat object.

     Usage:
     > loss = MSE()(y, target)
     > loss.zero_grad()
     > loss.backward()

     Note that backward does not propagate the gradient for the target.
    '''
    def __call__(self, y: Mat, target: Mat) -> Mat:
        self.args = [y]
        self.target = target
        diff = self.args[0].val - target.val
        mse_val = np.sum(diff**2) / (2 * y.rows)
        v = Mat(np.array([mse_val]))  # Create a scalar Mat object for loss
        v.creator = self
        return v


    def backward(self, s=1.):
        grad = (self.args[0].val - self.target.val) / self.args[0].rows
        self.args[0].backward(s * grad)



class Connection():
    '''
     lyr = Connection(from_nodes=1, to_nodes=1)

     Creates a layer of all-to-all connections.

     Inputs:
       from_nodes  number of nodes in source layer
       to_nodes    number of nodes in receiving layer
      
     Usage:
       lyr.W       # holds the connection weights (Mat object)
       lyr.b       # holds the biases (Mat object)
       lyr.params  # a list of learnable parameters, [W,b]
       z = lyr(h)  # projects activities h (Mat) through the
                   # connection to get input currents z (Mat)
                   # ie. z = h @ W + b
    '''
    def __init__(self, from_nodes=1, to_nodes=1):
        self.Nin = from_nodes    # number of nodes in input (lower) layer
        self.Nout = to_nodes     # number of nodes in higher layer

        # Random initial connection weights (W) and biases (b).
        self.W = Mat( np.random.randn(self.Nin, self.Nout) / np.sqrt(self.Nin) )
        self.b = Mat( np.zeros(self.Nout) )
        
        # Add W and b to the list of learnable parameters.
        self.params = [self.W, self.b]

    def __call__(self, x):
        Mul1 = Mul()
        Mul2 = Mul()

        pls = Plus()

        A = Mul1(x, self.params[0])
        C = Mat(np.ones((x.rows,1)))
        B = Mul2(C, self.params[1])

        return pls(A, B) 


class Network(object):
    '''
     net = Network()

     Creates a Network object, which represents a series of
     layers, and a cost (loss) function.

     Usage:
       net.lyr     # returns a list of Layer objects
       net.add_layer(L)  # appends the Layer L to the network
       y = net(x)  # feeds inputs x through the network
                   # x is a 2D numpy array or a Mat object
                   # returns y (Mat object)
       net.loss    # loss function (MatOperation object)
    '''
    def __init__(self):
        self.lyr = []
        self.loss = None   # the loss function to minimize (MatOperation)
        self.loss_history = []

    def add_layer(self, L):
        self.lyr.append(L)

    def __call__(self, x):
        for l in self.lyr:
            x = l(x)
        return x

    def parameters(self):
        '''
         params = net.parameters()
         
         Returns a list of Mat objects of the connection weights and
         biases in the network (from bottom to top).
         Note that only Connection layers have these parameters.
        '''
        params = []
        for l in self.lyr:
            if hasattr(l, 'params') and l.params:  # skips empty lists
                for p in l.params:
                    params.append(p)
        return params
    
    def learn(self, ds, lrate=1., epochs=10):
        '''
         net.learn(ds, lrate=1., epochs=10)

         Runs backprop on the network, training on the data from
         the Dataset object ds.

         Inputs:
           ds       a Dataset object
           lrate    learning rate
           epochs   number of epochs to run
        '''
        x = ds.inputs()
        t = ds.targets()
        for epoch in range(epochs):
            y = self(x)             # forward
            y = self(x)             # forward
            # Evaluate the loss
            loss = self.loss(y, t)
            # Reset gradient to zero
            loss.zero_grad()
            # Backpropagate the gradients
            loss.backward()
            # Update learnable network parameters (weights and biases)
            for p in self.parameters():
                p.val = p.val - lrate * p.grad

            self.loss_history.append(loss.val[0,0])

            # Report progress
            if epoch%50==0:
                print(f'{epoch}: cost = {loss()[0,0]}') 


