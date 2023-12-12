from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics.texture import Texture
from kivy.uix.camera import Camera
from kivy.lang import Builder
from jnius import autoclass
import numpy as np
import cv2
#import os
#from kivy.uix.label import Label
#from model import TensorFlowModel

# props to tibssy
# https://github.com/tibssy/Kivy-Android-Camera

CameraInfo = autoclass('android.hardware.Camera$CameraInfo')
CAMERA_INDEX = {'front': CameraInfo.CAMERA_FACING_FRONT, 'back': CameraInfo.CAMERA_FACING_BACK}
Builder.load_file("myapplayout.kv")


class AndroidCamera(Camera):
    resolution = (640, 480)
    index = CAMERA_INDEX['back']
    counter = 0

    def on_tex(self, *l):
        if self._camera._buffer is None:
            return None

        super(AndroidCamera, self).on_tex(*l)
        self.texture = Texture.create(size=np.flip(self.resolution), colorfmt='rgb')
        frame = self.frame_from_buf()
        self.frame_to_screen(frame)

    def frame_from_buf(self):
        w, h = self.resolution
        frame = np.frombuffer(self._camera._buffer.tostring(), 'uint8').reshape((h + h // 2, w))
        frame_bgr = cv2.cvtColor(frame, 93)
        if self.index:
            return np.flip(np.rot90(frame_bgr, 1), 1)
        else:
            return np.rot90(frame_bgr, 3)

    def frame_to_screen(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.putText(frame_rgb, str(self.counter), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        self.counter += 1
        flipped = np.flip(frame_rgb, 0)
        buf = flipped.tobytes()
        self.texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')

    def tensorflow_lite_object_detection(self):
        self.label_cnn.text = str("Dies ist ein Text")
        #model_to_predict = TensorFlowModel()
        #model_to_predict.load(os.path.join(os.getcwd(), 'model.tflite'))
        #np.random.seed(42)
        #x = np.array(np.random.random_sample((1, 28, 28)), np.float32)
        #y = model_to_predict.pred(x)
        # result should be
        # 0.01647118,  1.0278152 , -0.7065112 , -1.0278157 ,  0.12216613,
        # 0.37980393,  0.5839217 , -0.04283606, -0.04240461, -0.58534086
        #return Label(text=f'{y}')
        #self.label_cnn.text = str("Works")

class MyLayout(BoxLayout):
    pass


class MyApp(App):
    def build(self):
        return MyLayout()


if __name__ == '__main__':
    MyApp().run()
