import numpy as np

class special_functions():

    def __init__(self) -> None:
        
        return None
    
    def sigmoid (self, x):
        return 1/(1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)

class web_class():

    sg_Func = special_functions()

    def __init__(self) -> None:
        
        self.inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
        self.expected_output = np.array([[0],[1],[1],[0]])

        self.epochs = 10000
        self.lr = 0.1
        self.inputLayerNeurons, self.hiddenLayerNeurons, self.outputLayerNeurons = 2,2,1

        self.hidden_weights = np.random.uniform(size=(self.inputLayerNeurons,self.hiddenLayerNeurons))
        self.hidden_bias =np.random.uniform(size=(1,self.hiddenLayerNeurons))
        self.output_weights = np.random.uniform(size=(self.hiddenLayerNeurons,self.outputLayerNeurons))
        self.output_bias = np.random.uniform(size=(1,self.outputLayerNeurons))

        return None
    
    def check_weights(self):
        print("Initial hidden weights: ",end='')
        print(*self.hidden_weights)
        print("Initial hidden biases: ",end='')
        print(*self.hidden_bias)
        print("Initial output weights: ",end='')
        print(*self.output_weights)
        print("Initial output biases: ",end='')
        print(*self.output_bias) 
        return None

    def __Fr_Propagation__(self) -> None:
        self.hidden_layer_activation = np.dot(self.inputs, self.hidden_weights)
        self.hidden_layer_activation += self.hidden_bias
        self.hidden_layer_output = self.sg_Funcs.sigmoid(self.hidden_layer_activation)
        self.output_layer_activation = np.dot(self.hidden_layer_output, self.output_weights)
        self.output_layer_activation += self.output_bias
        self.predicted_output = self.sg_Func.sigmoid(self.output_layer_activation)
        return None
    
    def __Bc_Propagation__(self) -> None:

        error = self.expected_output - self.predicted_output
        d_predicted_output = error * self.sg_Func.sigmoid_derivative(self.predicted_output)
    
        error_hidden_layer = d_predicted_output.dot(self.output_weights.T)
        d_hidden_layer = error_hidden_layer * self.sg_Func.sigmoid_derivative(self.hidden_layer_output)

        return None

    def __W_update__(self):

        output_weights += self.hidden_layer_output.T.dot(self.d_predicted_output) * self.lr
        output_bias += np.sum(self.d_predicted_output,axis=0,keepdims=True) * self.lr
        hidden_weights += self.inputs.T.dot(self.d_hidden_layer) * self.lr
        hidden_bias += np.sum(self.d_hidden_layer,axis=0,keepdims=True) * self.lr

        return None
    
    def __Train_algth__(self) -> None:
        self.__Fr_Propagation__()
        self.__Bc_Propagation__()
        self.__W_update__()

        return None

    def learn_start(self, x = 0):
        if(x == 0):
            x = self.epochs
        
        for i in range(x):
            self.__Train_algth__()

    def result(self, x=0):
        print("Final hidden weights: ",end='')
        print(*self.hidden_weights)
        print("Final hidden bias: ",end='')
        print(*self.hidden_bias)
        print("Final output weights: ",end='')
        print(*self.output_weights)
        print("Final output bias: ",end='')
        print(*self.output_bias)

        print("\nOutput from neural network after 10,000 epochs: ",end='')
        print(*self.predicted_output)