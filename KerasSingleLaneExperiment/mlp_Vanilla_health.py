from keras.models import Sequential
from keras.layers import Dense,Input,Lambda, Activation
from KerasSingleLaneExperiment.LambdaLayers import add_node_layers
from keras.models import Model

def define_vanilla_model_MLP(num_vars,num_classes,hidden_units):
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

    # edge node
    edge = define_MLP_architecture_edge(img_input, hidden_units)

    # fog node 2
    fog2 = define_MLP_architecture_fog2(edge, hidden_units)

    # fog node 1
    fog1 = define_MLP_architecture_fog1(fog2, hidden_units)
    fog1 = Lambda(lambda x: x * 1,name="Cloud_Input")(fog1)
    # cloud node
    cloud = define_MLP_architecture_cloud(fog1, hidden_units, num_classes)

    model = Model(inputs=img_input, outputs=cloud)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def define_MLP_architecture_edge(img_input,hidden_units):
    edge_output = Dense(units=hidden_units,name="edge_output_layer",activation='relu')(img_input)
    return edge_output

def define_MLP_architecture_fog2(fog2_input,hidden_units):
    fog2 = Dense(units=hidden_units,name="fog2_input_layer",activation='relu')(fog2_input)
    fog2_output = Dense(units=hidden_units,name="fog2_output_layer",activation='relu')(fog2)
    return fog2_output

def define_MLP_architecture_fog1(fog1_input,hidden_units):
    fog1 = Dense(units=hidden_units,name="fog1_input_layer",activation='relu')(fog1_input)
    fog1 = Dense(units=hidden_units,name="fog1_layer_1",activation='relu')(fog1)
    fog1_output = Dense(units=hidden_units,name="fog1_output_layer",activation='relu')(fog1)
    return fog1_output

def define_MLP_architecture_cloud(cloud_input, hidden_units, num_classes):
    cloud = Dense(units=hidden_units,name="cloud_input_layer",activation='relu')(cloud_input)
    cloud = Dense(units=hidden_units,name="cloud_layer_1",activation='relu')(cloud)
    cloud = Dense(units=hidden_units,name="cloud_layer_2",activation='relu')(cloud)
    cloud = Dense(units=hidden_units,name="cloud_layer_3",activation='relu')(cloud)
    cloud_output = Dense(units=num_classes,activation='softmax',name = "output")(cloud)
    return cloud_output