import os
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.filechooser import FileChooserIconView
from kivy.graphics.texture import Texture
from image_dithering import ImageDithering

def open_image(name, img_scale=7):
    normalized_path = os.path.normpath(name)
    processed_img = ImageDithering.processing(normalized_path, img_scale)
    return processed_img

class ImageApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical')

        self.filechooser = FileChooserIconView(filters=['*.png', '*.jpg', '*.jpeg'])
        self.filechooser.size_hint = (1, 0.6)
        self.filechooser.path = 'C:/Users/giann/Pictures'
        self.layout.add_widget(self.filechooser)

        self.open_button = Button(text="Open Selected Image", size_hint=(1, 0.1))
        self.open_button.bind(on_release=self.open_selected_image)
        self.layout.add_widget(self.open_button)

        self.image_widget = Image(size_hint=(1, 0.3))
        self.layout.add_widget(self.image_widget)

        return self.layout

    def open_selected_image(self, instance):
        selection = self.filechooser.selection
        if selection:
            image_array = open_image(selection[0])
            if image_array is not None:
                texture = self.numpy_to_texture(image_array)
                self.image_widget.texture = texture

    def numpy_to_texture(self, img):
        texture = Texture.create(size=(img.shape[1], img.shape[0]), colorfmt='luminance')
        texture.blit_buffer(img.tobytes(), colorfmt='luminance', bufferfmt='ubyte')
        texture.flip_vertical()
        return texture

if __name__ == '__main__':
    ImageApp().run()
