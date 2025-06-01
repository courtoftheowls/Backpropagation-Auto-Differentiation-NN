#=======================================
#
#  unit tests
#
#=======================================
import pytest
from mat import *

#You need to include the new classes ReLU, Connection,... that you implemented in mat.py to be able to do testing

#===== test_mul_backward =====
def test_mul_grad_dim():

    # Test the backward method of the Mul class
    a = Mat(0.1*np.ones((3,2)))
    b = Mat(np.ones((2,4)))
    s = np.ones((3,4))
    y = Mul()(a,b)
    y.backward(s)

    assert a.grad.shape == (3,2), 'Mul.backward: grad wrong shape'
    assert b.grad.shape == (2,4), 'Mul.backward: grad wrong shape'

def test_mul_grad():

    # Test the backward method of the Mul class give correct grad values
    a = Mat(np.array([[3.,4]]))
    b = Mat(np.array([[1.],[-2]]))
    s = 0.1*np.ones((1,1))
    y = Mul()(a,b)
    y.backward(s)

    assert a.grad == pytest.approx(np.array([[0.1, -0.2]]), rel=1e-4), 'Mul.backward: grad incorrect values'
    assert b.grad == pytest.approx(np.array([[0.3], [0.4]]), rel=1e-4), 'Mul.backward: grad incorrect values'

test_mul_grad_dim
test_mul_grad()


#===== test_Connection =====
def test_call_shape():
    # Does the Connection.__call__ method work?
    cl1 = Connection(from_nodes=2, to_nodes=3)

    x = Mat(np.ones((4,2)))
    y = cl1(x)

    assert y.val.shape == (4,3), 'Connection.__call__ outputs wrong dimensions'
    
def test_call_val():
    # Does the Connection.__call__ method work?
    cl1 = Connection(from_nodes=2, to_nodes=1)
    cl1.W.val[0,0] = 0.5
    cl1.W.val[1,0] = 1.6
    cl1.b.val[0,0] = 0.1

    x = Mat(np.array([[2,-1.]]))
    y = cl1(x)

    assert y.val[0,0] == pytest.approx(-0.5, rel=1e-6), 'Connection.__call__ outputs wrong value'

test_call_shape()
test_call_val()

def test_mse():
    # Test forward computation
    y = Mat(np.array([[0.2, 0.3], [0.4, 0.5]]))  # Predicted values
    target = Mat(np.array([[0.1, 0.1], [0.1, 0.1]]))  # Target values

    # Hardcoded expected loss
    hardcoded_expected_loss = 0.075 

    mse = MSE()
    loss = mse(y, target)

    # Assert forward pass correctness
    assert np.isclose(loss.val[0, 0], hardcoded_expected_loss), f"Expected {hardcoded_expected_loss}, got {loss.val[0, 0]}"

    # Test backward computation
    loss.zero_grad()
    loss.backward()

    # Hardcoded expected gradient
    hardcoded_expected_grad = np.array([[0.05, 0.1], [0.15, 0.2]]) 
    assert np.allclose(y.grad, hardcoded_expected_grad), f"Expected gradient {hardcoded_expected_grad}, got {y.grad}"

    # Test edge case: y equals target
    y_equal_target = Mat(np.array([[0.5, 0.5], [0.5, 0.5]]))
    target_equal = Mat(np.array([[0.5, 0.5], [0.5, 0.5]]))
    loss_equal = mse(y_equal_target, target_equal)
    assert np.isclose(loss_equal.val[0, 0], 0), "Loss should be zero when y equals target"

    # Test edge case: all zeros
    y_zeros = Mat(np.zeros((2, 2)))
    target_zeros = Mat(np.zeros((2, 2)))
    loss_zeros = mse(y_zeros, target_zeros)
    assert np.isclose(loss_zeros.val[0, 0], 0), "Loss should be zero when both y and target are zeros"

test_mse()

#===== test_bgrad =====
def test_bgrad():

    net = Network()
    l0 = Connection(from_nodes=2, to_nodes=3)
    l1 = ReLU()
    net.add_layer(l0)
    net.add_layer(l1)
    net.loss = MSE()
    l0.W.val[:] = 1.
    l0.b.val[:] = 0.

    x = Mat(np.ones((4,2)))
    y = net(x)
    t = Mat(np.ones_like(y.val))
    loss = net.loss(y,t)
    loss.zero_grad()
    loss.backward()

    assert len(net.lyr[0].b.grad) == 1, 'Connection.b.grad is wrong dimensions'

test_bgrad()