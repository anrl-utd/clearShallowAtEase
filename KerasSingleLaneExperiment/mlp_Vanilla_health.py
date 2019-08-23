from keras.models import Sequential
from keras.layers import Dense,Input,Lambda, Activation
from KerasSingleLaneExperiment.LambdaLayers import add_node_layers
from keras.models import Model

def define_vanilla_model(num_vars,num_classes,hidden_units):
    """Define a normal neural network.
   ### Naming Convention
        ex: f2f1 = connection between fog node 2 and fog node 1
    ### Arguments
        num_vars (int): specifies number of variables from the data, used to determine input size.
        num_classes (int): specifies number of classes to be outputted by the model
        hidden_units (int): specifies number of hidden units per layer in network
      
    ### Returns
        Keras Model object
    """

    # IoT Node
    iot = define_mlp_architecture_IoT(num_vars)

    # edge node
    edge = define_mlp_architecture_edge(iot, hidden_units)

    # fog node 2
    fog2 = define_mlp_architecture_fog2(edge, hidden_units)

    # fog node 1
    fog1 = define_mlp_architecture_fog1(fog2, hidden_units)

    # cloud node
    cloud = define_mlp_architecture_cloud(fog1, hidden_units, num_classes)

    model = Model(inputs=iot, outputs=cloud)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def define_mlp_architecture_IoT(num_vars):
    return Input(shape = (num_vars,))

def define_mlp_architecture_edge(iot_output,hidden_units):
    edge_input = Dense(units=hidden_units,name="edge_output_layer")(iot_output)
    edge = Activation(activation='relu')(edge_input)
    edge_output = Lambda(lambda x: x * 1,name="F2_Input")(edge)
    return edge_output

def define_mlp_architecture_fog2(edge_output,hidden_units):
    fog2_input = Dense(units=hidden_units,name="fog2_input_layer")(edge_output)
    fog2 = Activation(activation='relu')(fog2_input)
    fog2 = Dense(units=hidden_units,name="fog2_output_layer")(fog2)
    fog2 = Activation(activation='relu')(fog2)
    fog2_output = Lambda(lambda x: x * 1,name="F1_Input")(fog2)
    return fog2_output

def define_mlp_architecture_fog1(fog2_output,hidden_units):
    fog1_input = Dense(units=hidden_units,name="fog1_input_layer")(fog2_output)
    fog1 = Activation(activation='relu')(fog1_input)
    fog1 = Dense(units=hidden_units,name="fog1_layer_1")(fog1)
    fog1 = Activation(activation='relu')(fog1)
    fog1 = Dense(units=hidden_units,name="fog1_output_layer")(fog1)
    fog1 = Activation(activation='relu')(fog1)
    fog1_output = Lambda(lambda x: x * 1,name="Cloud_Input")(fog1)
    return fog1_output

def define_mlp_architecture_cloud(fog1_output, hidden_units, num_classes):
    cloud_input = Dense(units=hidden_units,name="cloud_input_layer")(fog1_output)
    cloud = Activation(activation='relu')(cloud_input)
    cloud = Dense(units=hidden_units,name="cloud_layer_1")(cloud)
    cloud = Activation(activation='relu')(cloud)
    cloud = Dense(units=hidden_units,name="cloud_layer_2")(cloud)
    cloud = Activation(activation='relu')(cloud)
    cloud = Dense(units=hidden_units,name="cloud_layer_3")(cloud)
    cloud = Activation(activation='relu')(cloud)
    cloud_output = Dense(units=num_classes,activation='softmax',name = "output")(cloud)
    return cloud_output