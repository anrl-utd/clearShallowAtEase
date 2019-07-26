from keras.models import Sequential
from keras.layers import Dense,Input,Lambda, Activation
from KerasSingleLaneExperiment.LambdaLayers import add_node_layers
from keras.models import Model

def define_vanilla_model(num_vars,num_classes,hidden_units):
    """Define a normal neural network.
   ### Naming Convention
        ex: f1f2 = connection between fog node 1 and fog node 2
    ### Arguments
        num_vars (int): specifies number of variables from the data, used to determine input size.
        num_classes (int): specifies number of classes to be outputted by the model
        hidden_units (int): specifies number of hidden units per layer in network
      
    ### Returns
        Keras Model object
    """
    # naming convention:
    # ex: f1f2 = connection between fog node 1 and fog node 2

    connection_weight_ef2 = 1
    connection_weight_f2f1 = 1
    connection_weight_f1c = 1


    # define lambdas for multiplying node weights by connection weight
    multiply_weight_layer_f1f2 = Lambda((lambda x: x * connection_weight_ef2), name = "connection_weight_f1f2")
    multiply_weight_layer_f2f3 = Lambda((lambda x: x * connection_weight_f2f1), name = "connection_weight_f2f3")
    multiply_weight_layer_f1c = Lambda((lambda x: x * connection_weight_f1c), name = "connection_weight_f3c")

    # IoT Node
    IoT_node = Input(shape = (num_vars,))

    # edge node
    e = Dense(units=hidden_units,name="edge_output_layer")(IoT_node)
    e = Activation(activation='relu')(e)
    f1f2 = multiply_weight_layer_f1f2(e)
    connection_f2 = Lambda(add_node_layers,name="E_F2")([f1f2])

    # fog node 2
    f2 = Dense(units=hidden_units,name="fog2_input_layer")(connection_f2)
    f2 = Activation(activation='relu')(f2)
    f2 = Dense(units=hidden_units,name="fog2_output_layer")(f2)
    f2 = Activation(activation='relu')(f2)
    f2f1 = multiply_weight_layer_f2f3(f2)
    connection_f1 = Lambda(add_node_layers,name="F2_F1")([f2f1])

    # fog node 1
    f1 = Dense(units=hidden_units,name="fog1_input_layer")(connection_f1)
    f1 = Activation(activation='relu')(f1)
    f1 = Dense(units=hidden_units,name="fog1_layer_1")(f1)
    f1 = Activation(activation='relu')(f1)
    f1 = Dense(units=hidden_units,name="fog1_output_layer")(f1)
    f1 = Activation(activation='relu')(f1)
    f1c = multiply_weight_layer_f1c(f1)
    connection_cloud = Lambda(add_node_layers,name="F1_FC")([f1c])

    # cloud node
    cloud = Dense(units=hidden_units,name="cloud_input_layer")(connection_cloud)
    cloud = Activation(activation='relu')(cloud)
    cloud = Dense(units=hidden_units,name="cloud_layer_1")(cloud)
    cloud = Activation(activation='relu')(cloud)
    cloud = Dense(units=hidden_units,name="cloud_layer_2")(cloud)
    cloud = Activation(activation='relu')(cloud)
    cloud = Dense(units=hidden_units,name="cloud_layer_3")(cloud)
    cloud = Activation(activation='relu')(cloud)
    
    # one output layer
    output_layer = Dense(units=num_classes,activation='softmax',name = "output")(cloud)
    model = Model(inputs=IoT_node, outputs=output_layer)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model