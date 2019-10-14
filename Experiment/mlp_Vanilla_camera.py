from keras.models import Sequential
from keras.layers import Dense,Input,Lambda, Activation, add, Flatten
from Experiment.LambdaLayers import add_node_layers
from keras.models import Model

def define_vanilla_model_MLP(input_shape,
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
    img_input_1 = Input(shape = input_shape)    
    img_input_2 = Input(shape = input_shape)
    img_input_3 = Input(shape = input_shape)
    img_input_4 = Input(shape = input_shape)    
    img_input_5 = Input(shape = input_shape) 
    img_input_6 = Input(shape = input_shape) 
    
    input_edge1 = add([img_input_1,img_input_2])
    input_edge2 = img_input_3
    input_edge3 = add([img_input_4,img_input_5])
    input_edge4 = img_input_6

    # edge nodes
    edge1 = define_MLP_architecture_edge(input_edge1, hidden_units, "edge1_output_layer")
    edge2 = define_MLP_architecture_edge(input_edge2, hidden_units, "edge2_output_layer")
    edge3 = define_MLP_architecture_edge(input_edge3, hidden_units, "edge3_output_layer")
    edge4 = define_MLP_architecture_edge(input_edge4, hidden_units, "edge4_output_layer")

    # fog node 4
    fog4_input = Lambda(add_node_layers,name="node5_input")([edge2,edge3, edge4])
    fog4 = define_MLP_architecture_fog_with_two_layers(fog4_input, hidden_units,"fog4_output_layer","fog4_input_layer")

    # fog node 3
    fog3 = Lambda(lambda x: x * 1,name="node4_input")(edge1)
    fog3 = define_MLP_architecture_fog_with_one_layer(fog3, hidden_units, "fog3_output_layer")

    # fog node 2
    fog2_input = Lambda(add_node_layers,name="node3_input")([fog3, fog4])
    fog2 = define_MLP_architecture_fog_with_two_layers(fog2_input, hidden_units, "fog2_output_layer", "fog2_input_layer")

    # fog node 1
    fog1 = Lambda(lambda x: x * 1,name="node2_input")(fog2)
    fog1 = define_MLP_architecture_fog_with_two_layers(fog1, hidden_units, "fog1_output_layer", "fog1_input_layer")
    

    # cloud node
    cloud = Lambda(lambda x: x * 1,name="node1_input")(fog1)
    cloud = define_MLP_architecture_cloud(cloud, hidden_units, num_classes)

    model = Model(inputs=[img_input_1,img_input_2,img_input_3,img_input_4,img_input_5,img_input_6], outputs=cloud)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def define_MLP_architecture_edge(edge_input, hidden_units, output_layer_name):
    edge_output = Dense(units=hidden_units, name=output_layer_name, activation='relu')(edge_input)
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
    cloud = Flatten()(cloud)
    cloud_output = Dense(units=num_classes,activation='softmax',name = "output")(cloud)
    return cloud_output