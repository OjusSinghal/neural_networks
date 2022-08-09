# Comparing accuracy and efficiency of vannila neural networks on MNIST


#### 1. Training results without vectorization of backprop function:



|Layers	        |Batch Size |Eta    |Epochs	|Time/epoch |Accuracy   |
|---	        |---	    |---	|---	|---    	|---	    |
|[784, 32, 10]  |1  	    |3.0   	|10   	|35 	    |0.81  	    |
|[784, 32, 10] 	|10  	    |3.0   	|10   	|25   	    |0.95  	    |
|[784, 32, 10] 	|100	    |3.0   	|10   	|25   	    |0.94  	    |
|[784, 32, 10] 	|1000	    |3.0   	|10   	|22   	    |0.88  	    |
