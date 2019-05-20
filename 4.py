#-*-coding:utf-8-*-
import json
import tornado.web
import tornado.websocket
import tornado.httpserver
import tornado.ioloop
import tornado.options
import matplotlib.pyplot as plt  
import tensorflow as tf  
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
import  mnist_cnn as mnist_interence
import mnist_train as mnist_train
EVAL_INTERVAL_SECS = 10
BATCH_SIZE = 100
import time

from PIL import Image

mnist=input_data.read_data_sets('./mni_data', one_hot=True)
rgb=[]

class BasicHandler(tornado.web.RequestHandler):
    def get(self):
        self.render('write/index.html')

class Picture(object):#这个类进行消息的处理
    def Accept(self):
        pass

    def Leave(self,newer):
        pass

    def Getmessage(self,newer,message):#接收消息并存储到数组中
        rgb.clear()#清空数组以便用户多次操作
        temp=message.split(",")
        for i in range(len(temp)):#所有string转int，构成RGBA数组
            temp[i] = int(temp[i])
        for i in range(len(temp)):#对接收到的图片RGBA->RGB
            if (i+1)%4==0:
                pass
            else:
                rgb.append(255-temp[i])

        imgs = np.array(rgb, dtype=np.uint8)#list转变成array
        imgs=imgs.reshape((28, 28, 3))

        a=Image.fromarray(imgs)  
        # a.show()#显示图像，查看是否得到准确图片
        a.save("D:\\a.gif","GIF")#图片保存下来，接着再次读取，确保自己的图片格式准确
        
        with tf.Session() as sess:

            img = tf.read_file("D:\\a.gif") #读取图片
            img_data = tf.image.decode_jpeg(img, channels=3) #解码
            #img_data = sess.run(tf.image.decode_jpeg(img, channels=3))
            imge=tf.image.convert_image_dtype(img_data,dtype=tf.float32)
            img_data = sess.run(tf.image.rgb_to_grayscale(imge)) #灰度化
            plt.imshow(img_data[:,:,0], cmap='gray')
            plt.show()
            temp=img_data[:,:,0]#获得灰度化图片数组
            temp=temp.reshape((28,28,1))#灰度化图片数组形式转换

            with tf.Graph().as_default():#对灰度化图片进行识别，此部分根据test.py修改
                x = tf.placeholder(tf.float32, shape=[None,
                                                    mnist_interence.IMAGE_SIZE,
                                                    mnist_interence.IMAGE_SIZE,
                                                    mnist_interence.NUM_CHANNEL], name='x-input')
                xs = temp
                reshape_xs = np.reshape(xs, (-1, mnist_interence.IMAGE_SIZE,
                                            mnist_interence.IMAGE_SIZE,
                                            mnist_interence.NUM_CHANNEL))
                y = mnist_interence.interence(x,False,None)
                key=tf.argmax(y,1)
                variable_average = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
                val_to_restore = variable_average.variables_to_restore()
                saver = tf.train.Saver(val_to_restore)
                
                with tf.Session() as sess:
                    ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_PATH)
                    if ckpt and ckpt.model_checkpoint_path:
                        saver.restore(sess,ckpt.model_checkpoint_path)#加载模型
                        key1=sess.run(key,feed_dict={x:reshape_xs})#开始feed
                        get=key1[0]#获得识别结果
                        message2 = {
                            'function': 'num',
                            'message': str(get)
                            }
                        newer.write_message(json.dumps(message2))#返回识别结果

class webChat(tornado.websocket.WebSocketHandler):
    def open(self):
        self.application.Picture.Accept()#建立连接
        print("建立连接")

    def on_close(self):
        self.application.Picture.Leave(self)#删除连接，清空data数组中的图片
        print("脱离连接")

    def on_message(self, message):   #接收客户端的图片数组信息
        self.application.Picture.Getmessage(self, message)
        


class Application(tornado.web.Application):    
    def __init__(self):
        self.Picture=Picture()
        
        handlers = [
            (r'/', BasicHandler),
            (r'/webChat', webChat),
        ]

        settings = {
            'template_path': '手写识别',
            'static_path': 'static'
        }

        tornado.web.Application.__init__(self, handlers, **settings)

if __name__ == '__main__':
    tornado.options.parse_command_line()
    server = tornado.httpserver.HTTPServer(Application())
    server.listen(8000)
    tornado.ioloop.IOLoop.instance().start()