# Package-PhoneNumber-OCR
用于识别快递包裹上的收件人电话号码的OCR系统（手机号码图像识别神经网络算法模型）

图像样本为计算机生成的四种字体（黑体、微软雅黑、微软Pgothic、Arial Unicode MS）单字图像，进行了一定的高斯模糊，并加入噪点。

matlab读取图像并处理为矩阵格式，用于神经网络模型的训练。

simple_character_classifier.py为训练神经网络的python程序，输出model.h5文件为已训练完成的模型。

multi-character_recognation.py是最终用于电话号码识别的py文件，它将调用已经训练好的model.h5文件并使用滑窗法对连续数字图像进行识别，输出的为11位阿拉伯数字。
