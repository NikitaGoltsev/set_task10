import numpy as np

class Lost_Func():

    f = np.vectorize(lambda x,y: np.exp(x)/y)

    def ReLU(self, x):

        return np.maximum(0, x)

    def tanh(self, x):

        return np.tanh(x)

    def softMax(self, x):
        exponentsSum = np.sum(np.exp(x))
        return self.f(x,exponentsSum)

class dop_functions():

    def maxPool(self, x, kernelSize, step):
        kernelSize = int(kernelSize)
        inX, inY, inC = x.shape
        maxPoolResult = np.zeros((inX//kernelSize, inY//kernelSize, inC))

        for channel in range(inC):
            for v_shift in range(0, inX - kernelSize + 1, step):
                for h_shift in range(0, inY - kernelSize + 1, step):
                    maxPoolResult[v_shift//kernelSize][h_shift//kernelSize][channel] = np.max(x[v_shift:v_shift+kernelSize,h_shift:h_shift+kernelSize,channel])
        return maxPoolResult

    def pad(self, matr, pad1, pad2=None):
        if pad2 == None:
            pad2 = pad1
        x, y, c = matr.shape
        resMatr = np.zeros((x+pad1+pad2, y+pad1+pad2, c))
        resMatr[pad1:-pad2, pad1:-pad2] = matr
        return resMatr

    def convolution(self, x, kernel, step):
        if x.shape[2] != kernel.shape[2]:
            raise ValueError("Shapes must be same")
        
        inX, inY, inC = x.shape
        kX, kY, kC = kernel.shape

        convResult = np.zeros((inX-1, inY-1, x.shape[2]))
        for channel in range(inC):
            for v_shift in range(0, inX - kX + 1, step):
                for h_shift in range(0, inY - kY + 1, step):
                    convResult[v_shift][h_shift][channel] = np.sum(
                        x[v_shift:v_shift+kX,h_shift:h_shift+kY,channel] * kernel[:,:,channel]
                    )
        convResult = np.sum(convResult, axis=2)
        return (convResult)


class Layer():

    def __init__(self, size, func_of_l = []):

        self.weights = np.random.random(size)
        self.lossFuncs = func_of_l

        return None

    def __call__(self, inputs):

        return self.forward(inputs)
    
    def forward(self, inputs):

        result = inputs.dot(self.weights)
        for f in self.lossFuncs:
            result = f(result)

        return result

class FoldLayer():

    loss_f = Lost_Func()
    dop = dop_functions()

    def __init__(self, sizeIn, sizeOut, kernelSize, pad, step):
        self.pad = pad
        self.step = step
        self.sIx, self.sIy, self.sIc = sizeIn
        self.sOx, self.sOy, self.sOc = sizeOut
        self.activatation = self.loss_f.ReLU

        self.weights = []
        for _ in range(sizeOut[2]):
            self.weights.append(np.random.random((kernelSize, kernelSize, self.sIc)))
        self.weights = np.array(self.weights)


    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        ny, nx, nc = inputs.shape
    
        result = np.zeros((ny-1, nx-1, self.sOc)) 

        for ind, weight in enumerate(self.weights):
            applied = self.dop.convolution(inputs, weight, step=self.step)
            result[:, :, ind] = applied[:, :]
        
        return self.activatation(result)

class MaxPoolLayer():

    dop = dop_functions()

    def __init__(self, kernelSize, step):
        self.step = step
        self.kernelSize = kernelSize
    
    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        result = self.dop.maxPool(inputs, self.kernelSize, step=self.step)
        return result
    

# To load code from first zd
class First_zd():

    loss_f = Lost_Func()

    def __init__(self, x:int = 1) -> None:

        
        #np.random.seed(100)

        global Layer

        inputs = np.random.randint(0, 255, size=(1, 256))
        layer_1 = Layer((256, 64), [self.loss_f.ReLU])
        layer_2 = Layer((64, 16), [self.loss_f.ReLU, self.loss_f.tanh])
        layer_3 = Layer((16, 4), [self.loss_f.ReLU, self.loss_f.softMax])

        self.conveyorFirstTask = [layer_1, layer_2, layer_3]

        if(x == 1):
            result = inputs
            for Layer in self.conveyorFirstTask:
                result = Layer(result)
            
            print(result)
        return None
    
    def get_conv(self):

        return self.conveyorFirstTask

# Second zd

class Second_zd():

    def __init__(self, x:int = 1) -> None:
        
        inputs = np.random.randint(0, 255, size=(19,19,3))

        x1 = FoldLayer((19,19,3), (18,18,8), kernelSize=3, pad=0, step=1)
        x2 = MaxPoolLayer(2, 2)
        x3 = FoldLayer((9,9,8), (8,8,16), 3, 0, 1)
        x4 = MaxPoolLayer(2, 2)


        self.conveyorSecondTask = [x1, x2, x3, x4]
        if( x == 1):
            data = inputs
            print(data.shape)
            for layer in self.conveyorSecondTask:
                data = layer(data)
                print(data.shape)
        

        return None

    def get_conv(self):

        return self.conveyorSecondTask

class Thrd_Zd():
    #Here I need to add first and second classes to one
    #also I need to use functions from Lost_func
    fst_class = First_zd(x = 0)
    scn_class = Second_zd(x = 0)

    def __create_date__(self):

        self.date = np.random.randint(0, 255, size=(19,19,3))
        self.conveyorThirdTask = self.scn_class.get_conv() + [np.ravel] + self.scn_class.get_conv()
        return None

    def __init__(self) -> None:
        self.__create_date__()
        self.__out_cicle__()
        return None

    def __out_cicle__(self) -> None:
        try:
            for layer in self.conveyorThirdTask:
                print(self.date.shape)
                self.date = layer(self.date)
            print(self.date.shape)
            print(self.date)
        except ValueError:
            print('Problem with value')
            
        return None

#fst = First_zd()

#snd = Second_zd()

frd = Thrd_Zd()