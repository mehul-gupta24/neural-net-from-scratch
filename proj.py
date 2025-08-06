import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import time

class DeepNeuralNetwork():
    def __init__(self,sizes,activation='sigmoid'):
        self.sizes=sizes
        if activation=='relu':
            self.activation=self.relu
        elif activation=='sigmoid':
            self.activation=self.sigmoid
        else:
            raise ValueError("Activation function is currently not support, please use 'relu' or 'sigmoid' instead.")
        
        self.params=self.initialize()
        self.cache={}

    def relu(self,x,derivative=False):
        # relu(x)=max(0,x)
        # der(relu(x))=1 when x>0 else 0
        if derivative:
            x=np.where(x<0,0,x)
            x=np.where(x>=0,1,x)
            return x
        return np.maximum(x,0)
    
    # def sigmoid(self,x,derivative=False):
    #     if derivative:
    #         return np.exp(-x)/(np.exp(-x)+1)**2
    #     # return 1/(1+np.exp(-x))
    #     return np.where(x >= 0, 
    #             1 / (1 + np.exp(-x)), 
    #             np.exp(x) / (1 + np.exp(x)))

    def sigmoid(self, x, derivative=False):
        x = np.clip(x, -500, 500)  # avoid overflow
        s = np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))
        if derivative:
            return s * (1 - s)
        return s


    # def softmax(self,x):
    #     # softmax(x)=exp(x)/summation(exp(x))
    #     exps=np.exp(x-x.max())
    #     return exps/np.sum(exps,axis=0)

    def softmax(self, x):
        x_shifted = x - np.max(x, axis=0, keepdims=True)  # avoid overflow
        exps = np.exp(x_shifted)
        return exps / np.sum(exps, axis=0, keepdims=True)


    def initialize(self):
        #number of nodes in each layer = 784 - hidden - output(10)
        input_layer=self.sizes[0]
        hidden_layer=self.sizes[1]
        output_layer=self.sizes[2]
        params={
            "w1":np.random.randn(hidden_layer,input_layer)*np.sqrt(1/input_layer),
            "b1":np.zeros((hidden_layer,1)),
            "w2":np.random.randn(output_layer,hidden_layer)*np.sqrt(1/hidden_layer),
            "b2":np.zeros((output_layer,1)),
        }
        return params

    def initialize_momemtum_optimizer(self):
        momentum_opt={
            "w1":np.zeros(self.params["w1"].shape),
            "b1":np.zeros(self.params["b1"].shape),
            "w2":np.zeros(self.params["w2"].shape),
            "b2":np.zeros(self.params["b2"].shape)
        }
        return momentum_opt


    def feedForwardPass(self,x):
        # y=sgm(w*x+b)
        self.cache["X"]=x
        self.cache["Z1"]=np.matmul(self.params["w1"],self.cache["X"].T)+self.params["b1"]
        self.cache["A1"]=self.activation(self.cache["Z1"])
        self.cache["Z2"]=np.matmul(self.params["w2"],self.cache["Z1"])+self.params["b2"]
        # self.cache["A2"]=self.activation(self.cache["Z2"])
        self.cache["A2"] = self.softmax(self.cache["Z2"])  # instead of self.activation(...)
        return self.cache["A2"]

    def backPropogation(self,y,output):
        current_batch_size=y.shape[0]
        dZ2=output-y.T
        dW2=(1/current_batch_size)*(np.matmul(dZ2,self.cache["A1"].T))
        db2=(1/current_batch_size)*(np.sum(dZ2,axis=1,keepdims=True))

        dA1=np.matmul(self.params["w2"].T,dZ2)
        dZ1=dA1*(self.activation(self.cache["Z1"],derivative=True))
        dW1=(1/current_batch_size)*(np.matmul(dZ1,self.cache["X"]))
        db1=(1/current_batch_size)*(np.sum(dZ1,axis=1,keepdims=True))
        self.grads = {"w1": dW1, "b1": db1, "w2": dW2, "b2": db2}
        return self.grads

    def cross_entropy_loss(self,y,output):
        # L(y,y^)=-summation(ylog(y^))
        output = np.clip(output, 1e-9, 1 - 1e-9)
        l_sum=np.sum(np.multiply(y.T,np.log(output)))
        m=y.shape[0]
        l=-(l_sum)/m
        return l


    def optimize(self,l_rate=0.1,beta=.9):
        '''
            Stochatic Gradient Descent (SGD):
            θ^(t+1) <- θ^t - η∇L(y, ŷ)
            
            Momentum:
            v^(t+1) <- βv^t + (1-β)∇L(y, ŷ)^t
            θ^(t+1) <- θ^t - ηv^(t+1)
        '''
        if self.optimizer == "sgd":
            for key in self.params:
                self.params[key] = self.params[key] - l_rate * self.grads[key]
        elif self.optimizer == "momentum":
            for key in self.params:
                self.momemtum_opt[key] = (beta * self.momemtum_opt[key] + (1. - beta) * self.grads[key])
                self.params[key] = self.params[key] - l_rate * self.momemtum_opt[key]
        else:
            raise ValueError("Optimizer is currently not support, please use 'sgd' or 'momentum' instead.")


    def accuracy(self,y,output):
        return np.mean(np.argmax(y,axis=-1)==np.argmax(output.T,axis=-1))

    def train(self, x_train, y_train, x_test, y_test, epochs=10, batch_size=64, optimizer='momentum', l_rate=.1, beta=.9):
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        num_batches = -(-x_train.shape[0] // self.batch_size)

        if self.optimizer == 'momentum':
            self.momemtum_opt = self.initialize_momemtum_optimizer()

        # For plots
        self.train_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.test_accuracies = []

        start_time = time.time()
        template = "Epoch {}: {:.2f}s, train acc={:.2f}, train loss={:.2f}, test acc={:.2f}, test loss={:.2f}"

        for i in range(self.epochs):
            permutation = np.random.permutation(x_train.shape[0])
            x_train_shuffled = x_train[permutation]
            y_train_shuffled = y_train[permutation]

            for j in range(num_batches):
                begin = j * self.batch_size
                end = min(begin + self.batch_size, x_train.shape[0] - 1)
                x = x_train_shuffled[begin:end]
                y = y_train_shuffled[begin:end]

                output = self.feedForwardPass(x)
                grad = self.backPropogation(y, output)
                self.optimize(l_rate=l_rate, beta=beta)

            # Evaluate performance
            output = self.feedForwardPass(x_train)
            train_acc = self.accuracy(y_train, output)
            train_loss = self.cross_entropy_loss(y_train, output)

            output = self.feedForwardPass(x_test)
            test_acc = self.accuracy(y_test, output)
            test_loss = self.cross_entropy_loss(y_test, output)

            # Save for plotting
            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)
            self.train_accuracies.append(train_acc)
            self.test_accuracies.append(test_acc)

            print(template.format(i + 1, time.time() - start_time, train_acc, train_loss, test_acc, test_loss))


    def ClassificationReport(self):
        # After training:
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.test_losses, label="Test Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss per Epoch")
        plt.legend()
        plt.show()

        plt.plot(self.train_accuracies, label="Train Accuracy")
        plt.plot(self.test_accuracies, label="Test Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy per Epoch")
        plt.legend()
        plt.show()
        # Classification Report
        outputs = self.feedForwardPass(x_test)
        y_pred = np.argmax(outputs, axis=0)
        y_true = np.argmax(y_test, axis=1)

        print(classification_report(y_true, y_pred))

        # Individual scores
        print("Macro F1 Score: ", f1_score(y_true, y_pred, average='macro'))
        print("Micro F1 Score: ", f1_score(y_true, y_pred, average='micro'))
        print("Recall Score: ", recall_score(y_true, y_pred, average='macro'))
        print("Precision Score: ", precision_score(y_true, y_pred, average='macro'))

        # Plot
        plt.figure(figsize=(8,6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()


def show_images(image,num_row,num_col):
    image_size=int(np.sqrt(image.shape[-1]))
    image=np.reshape(image,(image.shape[0],image_size,image_size))
    fig,axes=plt.subplots(num_row,num_col,figsize=(1.5*num_col,2*num_row))
    for i in range(num_row*num_col):
        # ax=axes[i//num_col,i%num_col]
        ax=axes[i]
        ax.imshow(image[i],cmap='gray',vmin=0,vmax=1)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def one_hot(x, k, dtype=np.float32):
    x = np.array(x)           # convert Series to ndarray
    x = x.reshape(-1)         # flatten if needed
    return np.eye(k, dtype=dtype)[x]

def main():
    mnist_data=fetch_openml("mnist_784")
    # mnist_data = fetch_openml("Fashion-MNIST")

    x=mnist_data["data"].astype('float32').to_numpy()
    y=mnist_data["target"].astype('int32').to_numpy()

    x/=255.0 #normalize


    # one hot encode labels
    num_labels=10
    examples=y.shape[0]
    y_new=one_hot(y,num_labels)
    # print(examples)

    train_size=60000
    test_size=x.shape[0]-train_size #test_size=10000

    x_train,x_test=x[:train_size],x[train_size:]
    y_train,y_test=y_new[:train_size],y_new[train_size:]

    shuffle_index=np.random.permutation(train_size)
    # print(shuffle_index)
    x_train,y_train=x_train[shuffle_index],y_train[shuffle_index]

    # print("Training data: {} {}".format(x_train.shape, y_train.shape))
    # print("Test data: {} {}".format(x_test.shape, y_test.shape))
    # show_images(x_train, 5, 5)
    return x_train,y_train,x_test,y_test


if __name__ == "__main__":

    x_train,y_train,x_test,y_test=main()
    # Sigmoid + Momentum
    dnn = DeepNeuralNetwork(sizes=[784,6, 10], activation='sigmoid')
    dnn.train(x_train, y_train, x_test, y_test,epochs=10, batch_size=128, optimizer='momentum', l_rate=0.1, beta=0.9)
    # dnn.ClassificationReport()

    # Relu + SGD
    # dnn = DeepNeuralNetwork(sizes=[784, 64, 10], activation='relu')
    # dnn.train(x_train, y_train, x_test, y_test,epochs=10, batch_size=128, optimizer='sgd', l_rate=0.05,beta=0.9)


    # checking updated weights and bias
    # for key, value in dnn.params.items():
    #     print(f"{key} shape: {value.shape}")
    #     print(value)

    # samples = x_test[:5]
    # outputs = dnn.feedForwardPass(samples)
    # predictions = np.argmax(outputs, axis=0)
    # actual = np.argmax(y_test[:5], axis=1)

    # print("Predicted:", predictions)
    # print("Actual:", actual)