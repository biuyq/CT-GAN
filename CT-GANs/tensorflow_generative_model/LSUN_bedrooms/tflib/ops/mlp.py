import tflib as lib
import tflib.ops.linear

import tensorflow as tf

def _ReLULayer(name, input_dim, output_dim, inputs):
    output = lib.ops.linear.Linear(
        name+'.Linear',
        input_dim=input_dim,
        output_dim=output_dim,
        inputs=inputs,
        initialization='glorot_he'
    )

    # output = tf.nn.relu(output)
    # output = tf.tanh(output)

    return output

def MLP(name, input_dim, hidden_dim, output_dim, n_layers, inputs):
    if n_layers < 3:
        raise Exception("An MLP with <3 layers isn't an MLP!")

    output = _ReLULayer(
        name+'.Input',
        input_dim=input_dim,
        output_dim=hidden_dim,
        inputs=inputs
    )

    for i in xrange(1,n_layers-2):
        output = _ReLULayer(
            name+'.Hidden'+str(i),
            input_dim=hidden_dim,
            output_dim=hidden_dim,
            inputs=output
        )

    # output = tf.stop_gradient(output)

    return lib.ops.linear.Linear(
        name+'.Output', 
        hidden_dim,
        output_dim, 
        output,
        initialization='glorot'
    )