'''
2.回答以下理论题目?

1. Compared to FNN, what is the biggest advantage of CNN?
Ans:减少神经网络中参数的个数

2. Suppose your input is a 100 by 100 gray image,
   and you use a convolutional layer with 50 filters that are each 5x5.
   How many parameters does this hidden layer have (including the bias parameters)?
Ans:(5*5+1)*50 = 1300


3. What are "local invariant" and "parameter sharing" ?
Ans:"local invariant" 表示平移不变性，图像中的目标不管被移动到图片的哪个位置，得到的结果（标签）应该是相同的；
    "parameter sharing"表示参数共享，

    在卷积神经网络中，卷积核内的一个卷积核(滤波器)用于提取一个特征(输入数据的一个维度)，
    而输入数据具有多个特征(维度)的话，就会有很多个卷积核，那么在这一层的卷积层中就会有“参数爆炸”的情况。
    如果一层中每个卷积核提取特定的特征，忽略了数据的局部相关性。
    而参数共享的作用在于，每个特征具有平移不变性，同一个特征可以出现在出现在不同数据的不同位置，可以用同一个卷积核来提取这一特征。
    而且利用数据的局部相关性，通过权值共享，一个卷积层共享一个卷积核，减少了卷积层上的参数。
    通过加深神经网络的层数(深度网络)，每一层卷积层使用不同的卷积核，从而达到提取尽可能多特征的目的。

    一方面，重复单元能够对特征进行识别，而不考虑它在可视域中的位置。
    另一方面，权值共享使得我们能更有效的进行特征抽取，因为它极大的减少了需要学习的自由变量的个数。

4. Why we use batch normalization ?
Ans:基本思想：因为深层神经网络在做非线性变换前的激活输入值随着网络深度加深或者在训练过程中，
    其分布逐渐发生偏移或者变动，之所以训练收敛慢，一般是整体分布逐渐往非线性函数的取值区间的上下限两端靠近
    （对于Sigmoid函数来说，意味着激活输入值WU+B是大的负值或正值），所以这导致反向传播时低层神经网络的梯度消失，
    这是训练深层神经网络收敛越来越慢的本质原因，
    而BN就是通过一定的规范化手段，把每层神经网络任意神经元这个输入值的分布强行拉回到均值为0方差为1的标准正态分布，
    其实就是把越来越偏的分布强制拉回比较标准的分布，这样使得激活输入值落在非线性函数对输入比较敏感的区域，
    这样输入的小变化就会导致损失函数较大的变化，让梯度变大，避免梯度消失问题产生，
    而且梯度变大意味着学习收敛速度快，能大大加快训练速度。

　　对于每个隐层神经元，把逐渐向非线性函数映射后向取值区间极限饱和区靠拢的输入分布强制拉回到均值为0方差为1的比较标准的正态分布，
    使得非线性变换函数的输入值落入对输入比较敏感的区域，以此避免梯度消失问题。
    因为梯度一直都能保持比较大的状态，所以很明显对神经网络的参数调整效率比较高。

    另外，并不是每次训练都要是高斯分布，引入γ、β参数，可以使分布进行偏移

5. What problem does dropout try to solve ?
Ans:使用部分神经元进行训练，防止over fitting的发生


6. Is the following statement correct and why ?
   "Because pooling layers do not have parameters, they do not affect the backpropagation(derivatives) calculation"
Ans:这句话不对，池化是不需要参数就能进行降维。但是不论是mean pooling还是max pooling都会对反向传播有影响的。

    mean pooling的前向传播就是把一个patch中的值求取平均来做pooling，
    那么反向传播的过程也就是把某个元素的梯度等分为n份分配给前一层，这样就保证池化前后的梯度（残差）之和保持不变

    max pooling也要满足梯度之和不变的原则，max pooling的前向传播是把patch中最大的值传递给后一层，而其他像素的值直接被舍弃掉。
    那么反向传播也就是把梯度直接传给前一层某一个像素，而其他像素不接受梯度，也就是为0
    max pooling操作需要记录下池化操作时到底哪个像素的值是最大，也就是max id

'''