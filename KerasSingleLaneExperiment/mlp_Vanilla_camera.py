from keras.models import Sequential
from keras.layers import Dense,Input,Lambda, Activation
from KerasSingleLaneExperiment.LambdaLayers import add_node_layers
from keras.models import Model

def define_vanilla_model_MLP(num_vars,
                             num_classes,
                             hidden_units = 32):
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

    # IoT Node (input image)
    img_input = Input(shape = (num_vars,))
    # Brian: I think we need something like 4 different 'img_input's, so I named them img_input_1 through img_input_4

    # edge nodes
    edge1 = define_MLP_architecture_edge(img_input_1, hidden_units, "edge1_output_layer")
    edge2 = define_MLP_architecture_edge(img_input_2, hidden_units, "edge2_output_layer")
    edge3 = define_MLP_architecture_edge(img_input_3, hidden_units, "edge3_output_layer")
    edge4 = define_MLP_architecture_edge(img_input_4, hidden_units, "edge4_output_layer")

    # fog node 4
    fog4_input = Lambda(add_node_layers,name="fog4_input")([edge2,edge3, edge4])
    fog4 = define_MLP_architecture_fog_with_two_layers(fog4_input, hidden_units,"fog4_output_layer","fog4_input_layer")

    # fog node 3
    fog3 = define_MLP_architecture_fog_with_one_layer(edge1, hidden_units, "fog3_output_layer")

    # fog node 2
    fog2_input = Lambda(add_node_layers,name="fog2_input")([fog3, fog4])
    fog2 = define_MLP_architecture_fog_with_two_layers(fog2_input, hidden_units, "fog2_output_layer", "fog2_input_layer")

    # fog node 1
    fog1 = define_MLP_architecture_fog_with_two_layers(fog2, hidden_units, "fog1_output_layer", "fog1_input_layer")

    # cloud node
    cloud = define_MLP_architecture_cloud(fog1, hidden_units, num_classes)

    model = Model(inputs=img_input, outputs=cloud)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def define_MLP_architecture_edge(img_input, hidden_units, output_layer_name):
    edge_output = Dense(units=hidden_units, name=output_layer_name, activation='relu')(img_input)
    return edge_output

def define_MLP_architecture_fog_with_two_layers(fog_input, hidden_units, output_layer_name, input_layer_name):
    fog = Dense(units=hidden_units,name=input_layer_name ,activation='relu')(fog_input)
    fog_output = Dense(units=hidden_units,name=output_layer_name, activation='relu')(fog)
    return fog_output

def define_MLP_architecture_fog_with_one_layer(fog_input, hidden_units, output_layer_name):
    fog_output = Dense(units=hidden_units,name=output_layer_name ,activation='relu')(fog_input)
    return fog_output

def define_MLP_architecture_cloud(cloud_input, hidden_units, num_classes):
    cloud = Dense(units=hidden_units,name="cloud_input_layer",activation='relu')(cloud_input)
    cloud = Dense(units=hidden_units,name="cloud_layer_1",activation='relu')(cloud)
    cloud = Dense(units=hidden_units,name="cloud_layer_2",activation='relu')(cloud)
    cloud_output = Dense(units=num_classes,activation='softmax',name = "output")(cloud)
    return cloud_output