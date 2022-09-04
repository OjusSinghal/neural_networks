# Comparing accuracy and efficiency of vannila neural networks on MNIST


#### 1. Training results without vectorization of backprop function:



|Layers	        |Batch Size |Eta    |Epochs	|Time/epoch |Accuracy   |Activation funct   |
|---	        |---	    |---	|---	|---    	|---	    |---                |
|[784, 32, 10]  |1  	    |0.01  	|10   	|35 	    |0.91  	    |Sigmoid            |
|[784, 32, 10] 	|10  	    |1.0   	|10   	|25   	    |0.93  	    |Sigmoid            |
|[784, 32, 10] 	|100	    |5.0   	|10   	|25   	    |0.93  	    |Sigmoid            |
|[784, 32, 10] 	|1000	    |10.0  	|10   	|22   	    |0.90  	    |Sigmoid            |


#### 2. Training results with vectorization of backprop function:


|Layers	        |Batch Size |Eta    |Epochs	|Time/epoch |Accuracy   |Activation funct   |
|---	        |---	    |---	|---	|---    	|---	    |---                |
|[784, 32, 10]  |1  	    |0.01  	|10   	|25 	    |0.90  	    |TanH               |
|[784, 32, 10] 	|10  	    |0.2   	|10   	|1   	    |0.90  	    |TanH               |
|[784, 32, 10] 	|100	    |3.0   	|10   	|1   	    |0.91  	    |TanH               |
|[784, 32, 10] 	|1000	    |5.0   	|10   	|1   	    |0.80  	    |TanH               |

