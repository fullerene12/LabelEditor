import sys
from qtpy.QtWidgets import QApplication, QMainWindow, QInputDialog
from qtpy.QtCore import Slot
from qtpy import uic
from logged_quantity import LoggedQuantity, FileLQ
import cv2
import pyqtgraph as pg
import numpy as np
import pandas as pd
from frame_check import find_bad_frame
import xml.etree.cElementTree as ET


class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.title = 'LabelChecker'

        # load custom UI
        self.ui = uic.loadUi('main.ui', self)

        # set up display
        self.video_layout = pg.GraphicsLayoutWidget()
        self.ui.video_groupBox.layout().addWidget(self.video_layout)
        self.video_view = pg.ViewBox()
        self.video_layout.addItem(self.video_view)

        self.video_image = pg.ImageItem()
        self.video_view.addItem(self.video_image)
        self.video_image.setOpts(axisOrder='row-major')

        self.scatter = pg.ScatterPlotItem()
        self.video_view.addItem(self.scatter)

        # setup and load all logged quantities

        # likelihood threshold
        self.likelihood = LoggedQuantity(name='likelihood', dtype=float, ro=False, initial=0.95, vmin=0, vmax=1)
        self.likelihood.connect_to_widget(self.ui.likelihood_doubleSpinBox)

        # total number of frames
        self.total_frame = LoggedQuantity(name='total_frame', dtype=int, ro=True, initial=0, vmin=0)
        self.total_frame.connect_to_widget(self.ui.total_frame_doubleSpinBox)

        # current displayed frame
        self.current_frame = LoggedQuantity(name='current_frame', dtype=int, initial=0, ro=True,
                                            vmin=0, vmax=0)
        self.current_frame.connect_to_widget(self.ui.current_frame_doubleSpinBox)
        self.current_frame.connect_to_widget(self.ui.current_frame_horizontalSlider)
        self.current_frame.updated_value.connect(self.load_frame)

        # total number of bad frames
        self.total_bad_frame = LoggedQuantity(name='total_bad_frame', dtype=int, ro=True, initial=0, vmin=0)
        self.total_bad_frame.connect_to_widget(self.total_bad_frame_doubleSpinBox)

        # current displayed bad frame
        self.current_bad_frame = LoggedQuantity(name='current_bad_frame', dtype=int, initial=0, ro=True,
                                            vmin=0, vmax=0)
        self.current_bad_frame.connect_to_widget(self.ui.current_bad_frame_doubleSpinBox)
        self.current_bad_frame.connect_to_widget(self.ui.current_bad_frame_horizontalSlider)
        self.current_bad_frame.updated_value.connect(self.load_bad_frame)

        # Video File Path
        self.video_file_path = FileLQ(name='video_file_path',default_dir='./')
        self.video_file_path.connect_to_browse_widgets(self.ui.video_file_path_lineEdit, self.ui.set_dir_pushButton)
        self.video_file_path.update_value('./')
        # position file suffix
        self.position_suffix = LoggedQuantity(name='position_suffix', dtype=str,
                                              initial='DeepCut_resnet101_trackingledsAug13shuffle1_500000')
        self.position_suffix.connect_to_widget(self.ui.position_suffix_plainTextEdit)

        # load_position flag
        self.auto_path = LoggedQuantity(name='auto_path', dtype=bool, initial=True)
        self.auto_path.connect_to_widget(self.ui.auto_path_checkBox)
        self.load_position = LoggedQuantity(name='load_position', dtype=bool, initial=True)
        self.load_position.connect_to_widget(self.ui.load_position_checkBox)

        # connect button actions
        self.ui.load_pushButton.clicked.connect(self.load)
        self.ui.close_video_pushButton.clicked.connect(self.close_video)
        self.ui.next_frame_pushButton.clicked.connect(self.next_frame)
        self.ui.next_bad_frame_pushButton.clicked.connect(self.next_bad_frame)
        self.ui.clear_pushButton.clicked.connect(self.clear)
        self.ui.extrapolate_pushButton.clicked.connect(self.extrapolate)
        self.ui.actionSave_Settings.triggered.connect(self.save_settings)
        self.ui.actionLoad_Settings.triggered.connect(self.load_settings)

        # declare other attributes
        self.video = None
        self.click_proxy = None
        self.data_path = ''
        self.data_set = None
        self.num_label = 0
        self.current_label_id = 0
        self.bad_frames = None
        self.pens = None

        self.show()

        try:
            self.load_settings()
        except Exception as ex_msg:
            print(ex_msg)

    def load(self):
        try:
            # load video file
            self.video = Video(self.video_file_path.value)
            self.total_frame.update_value(self.video.total_frames)
            self.current_frame.change_min_max(vmin=0, vmax=self.video.total_frames-1)
            self.current_frame.change_readonly(ro=False)

            if self.load_position.value:
                # connect click event to set label method
                self.click_proxy = pg.SignalProxy(self.video_image.scene().sigMouseClicked, delay=0,
                                                  slot=self.set_label)

                # load coordinates for labels
                if self.video_file_path.value[-4:] == 'h264':
                    self.data_path = self.video_file_path.value[:-5] + self.position_suffix.value + '.h5'
                elif self.video_file_path.value[-3:] == 'mp4':
                    self.data_path = self.video_file_path.value[:-4] + self.position_suffix.value + '.h5'
                elif self.video_file_path.value[-3:] == 'avi':
                    self.data_path = self.video_file_path.value[:-4] + self.position_suffix.value + '.h5'
                else:
                    self.data_path = self.video_file_path.value.split('.')[0] + self.position_suffix.value + '.h5'
                self.data_set = pd.read_hdf(self.data_path)
                self.num_label = self.data_set[self.position_suffix.value].keys().levshape[0]
                self.labels = list()
                for i in range(self.num_label):
                    self.labels.append(self.data_set[self.position_suffix.value].keys()[i * 3][0])
                self.data_set[self.position_suffix.value].keys()[self.current_label_id * 3][0]

                # generate a list for pens to draw the scatter plot
                self.pens = list()

                for i in range(self.num_label):
                    pen = pg.mkPen(color=pg.mkColor(i))
                    self.pens.append(pen)

                # check for bad frames and load the information
                self.ui.progressBar.setValue(50)
                self.bad_frames = find_bad_frame(self.data_set, likelihood=self.likelihood.value,
                                                 check_progress=True, progress_bar=self.ui.progressBar)
                self.total_bad_frame.update_value(len(self.bad_frames))
                self.current_bad_frame.change_min_max(vmin =0, vmax=self.total_bad_frame.value-1)
                self.current_bad_frame.change_readonly(ro=False)

                # counter for alternating highlighted label
                self.current_label_id = 0

                self.ui.extrapolate_pushButton.setEnabled(True)
                self.ui.clear_pushButton.setEnabled(True)
                self.ui.next_bad_frame_pushButton.setEnabled(True)
                self.ui.smooth_and_close_pushButton.setEnabled(True)

            self.ui.next_frame_pushButton.setEnabled(True)
            self.ui.load_pushButton.setEnabled(False)
            self.ui.close_video_pushButton.setEnabled(True)
            self.likelihood.change_readonly(ro=True)
            self.video_file_path.change_readonly(ro=True)
            self.position_suffix.change_readonly(ro=True)
            self.load_position.change_readonly(ro=True)

            # refresh view box
            self.current_frame.update_value(1)
            self.ui.progressBar.setValue(100)

        except Exception as ex_msg:
            print(ex_msg)

            # abort and clear if any error occour
            self.close_video()

    def close_video(self):
        self.load_position.change_readonly(ro=False)
        self.ui.extrapolate_pushButton.setEnabled(False)
        self.ui.clear_pushButton.setEnabled(False)
        self.ui.next_bad_frame_pushButton.setEnabled(False)
        self.ui.smooth_and_close_pushButton.setEnabled(False)

        self.ui.next_frame_pushButton.setEnabled(False)
        self.ui.load_pushButton.setEnabled(True)
        self.ui.close_video_pushButton.setEnabled(False)
        self.likelihood.change_readonly(ro=False)
        self.video_file_path.change_readonly(ro=False)
        self.position_suffix.change_readonly(ro=False)

        try:
            # clear all video contents
            self.total_frame.update_value(0)
            self.total_bad_frame.update_value(0)
            self.current_frame.update_value(0)
            self.current_bad_frame.update_value(0)
            self.current_frame.change_min_max(vmin=0, vmax=0)
            self.current_bad_frame.change_min_max(vmin=0, vmax=0)
            self.current_frame.change_readonly(ro=True)
            self.current_bad_frame.change_readonly(ro=True)
            self.scatter.clear()
            self.video_image.clear()
            self.video.close()
            self.video = None

            # clear data contents and view box click binding
            self.data_path = ''
            self.num_label = 0
            self.current_label_id = 0
            self.click_proxy = None
            self.data_set = None
            self.bad_frames = None
            self.pens = None
            self.ui.progressBar.setValue(0)

        except Exception as ex_msg:
            print(ex_msg)

    def next_frame(self):
        try:
            self.current_frame.update_value(self.current_frame.value+1)
        except Exception as ex_msg:
            print(ex_msg)

    def next_bad_frame(self):
        try:
            self.current_bad_frame.update_value(self.current_bad_frame.value+1)
        except Exception as ex_msg:
            print(ex_msg)

    @Slot(float)
    def load_frame(self, i):
        if i>0:
            i = int(i)
            try:
                # load, display frame and draw label
                frame = self.video.read_frame(i)
                self.video_image.setImage(frame)
                if self.load_position.value:
                    self.load_scatter(i)
            except Exception as ex_msg:
                print(ex_msg)

    @Slot(float)
    def load_bad_frame(self, i):
        if i>0:
            i = int(i)
            try:
                frame_id = self.bad_frames[i]
                self.current_frame.update_value(frame_id)
                if self.load_position.value:
                    self.load_scatter(frame_id)
            except Exception as ex_msg:
                print(ex_msg)

    def load_scatter(self, i, range_limit=False, label_range=3):
        if range_limit:
            num_label = label_range
        else:
            num_label = self.num_label
        # create buffer for label coordinates
        try:
            x_array = np.zeros(shape=(num_label,))
            y_array = np.zeros(shape=(num_label,))

            # load label coordinates
            for j in range(num_label):
                label = self.labels[j]
                x_array[j] = self.data_set[self.position_suffix.value][label]['x'][i]
                y_array[j] = self.data_set[self.position_suffix.value][label]['y'][i]

            # draw labels
            self.scatter.setData(x=x_array, y=y_array, pen=self.pens[0:num_label])
        except Exception as ex_msg:
            print(ex_msg)

    def clear(self):
        self.scatter.clear()
        frame = self.video.read_frame(self.current_frame.value)
        self.video_image.setImage(frame)
        self.current_label_id = 0

    def set_label(self, event):
        # get x and y scene position from clicking
        spx = event[0].pos().x()
        spy = event[0].pos().y()

        # pyqtgraph scene position is not correctly aligned,
        # 10 must be added to the position to compensate for the drift
        view_pos = self.video_image.mapFromScene(pg.Point(spx+10, spy+10))

        # transform position from viewbox coordinate to image coordinate
        video_image_pos = self.video_image.mapFromView(view_pos)
        x = video_image_pos.x()
        y = video_image_pos.y()

        # highlight label
        label = self.labels[self.current_label_id]

        # modify the coordinate of highlighted label, and mark likelihood as 2 to show human modification
        self.data_set.loc[self.current_frame.value, self.position_suffix.value].loc[label, 'x'] = x
        self.data_set.loc[self.current_frame.value, self.position_suffix.value].loc[label, 'y'] = y
        self.data_set.loc[self.current_frame.value, self.position_suffix.value].loc[label, 'likelihood'] = 2

        # save new labeled data
        self.save_data_set()

        # reload image and new labels, then highlight the next label
        self.load_scatter(self.current_frame.value, range_limit=True, label_range=self.current_label_id + 1)
        self.current_label_id += 1
        self.current_label_id %= self.num_label

    def save_data_set(self):
        self.data_set.to_hdf(self.data_path, key=self.position_suffix.value, mode='w')

    def extrapolate(self):
        text_input, ok = QInputDialog.getText(self, 'Extrapolation',
                                              'Please input the start and end of extrapolation (e.g. 10-15)')
        if ok:
            try:
                start_id = int(text_input.split('-')[0])
                end_id = int(text_input.split('-')[1])
                self.perform_extrapolation(start_id,end_id)
            except Exception as ex_msg:
                print(ex_msg)

    def perform_extrapolation(self, start_frame, end_frame):
        array_length = end_frame - start_frame + 1

        for i in range(self.num_label):
            label = self.labels[i]
            start_x = self.data_set[self.position_suffix.value][label]['x'][start_frame]
            start_y = self.data_set[self.position_suffix.value][label]['y'][start_frame]
            end_x = self.data_set[self.position_suffix.value][label]['x'][end_frame]
            end_y = self.data_set[self.position_suffix.value][label]['y'][end_frame]
            xs = np.linspace(start_x,end_x,array_length)
            ys = np.linspace(start_y, end_y, array_length)
            self.data_set.loc[start_frame:end_frame,
                              self.position_suffix.value].loc[start_frame:end_frame,
                                                              label].loc[start_frame:end_frame, 'x'] = xs
            self.data_set.loc[start_frame:end_frame,
                              self.position_suffix.value].loc[start_frame:end_frame,
                                                              label].loc[start_frame:end_frame, 'y'] = ys
            self.data_set.loc[start_frame:end_frame,
                              self.position_suffix.value].loc[start_frame:end_frame,
                                                              label].loc[start_frame:end_frame, 'likelihood'] = 2

        self.save_data_set()

    def save_settings(self):
        try:
            file_name = 'settings.xml'
            doc = ET.Element('label_editor_lettings')
            item_list = dir(self)
            for name in item_list:
                item = getattr(self, name)
                if str(type(item)) == "<class 'logged_quantity.LoggedQuantity'>":
                    if not item.ro:
                        ET.SubElement(doc, 'setting', name=item.name, value=str(item.value))

            ET.SubElement(doc, 'setting', name='video_file_path', value=self.video_file_path.value)

            tree = ET.ElementTree(doc)
            tree.write(file_name)
        except Exception as ex_msg:
            print(ex_msg)

    def load_settings(self):
        try:
            file_name = 'settings.xml'
            settings = ET.parse(file_name).getroot()
            for setting in settings:
                if hasattr(self, setting.attrib['name']):
                    value = setting.attrib['value']
                    item = getattr(self, setting.attrib['name'])
                    item.update_value(value)
        except Exception as ex_msg:
            print(ex_msg)

class Video(object):

    def __init__(self, file_name):
        self.file_name = file_name
        self.cap = cv2.VideoCapture(self.file_name)

        # get number of frames, 7 is the propID for number of frames
        if self.file_name[-4:] == 'h264':
            try:
                meta_name = self.file_name[:-5] + '.npy'
                vshape = np.load(meta_name)
                self.num_frame = vshape[0]
            except Exception:
                self.num_frame = 0
                print('Could not find metatdata file '+meta_name)

        else:
            self.num_frame = self.cap.get(7)

    def read_frame(self, i):
        # 1 is the propID for current frame
        self.cap.set(1, i)
        ret, frame = self.cap.read()
        if ret == 1:
            return frame[:, :, 2::-1]
        else:
            raise Exception('Bad Frame')

    @property
    def total_frames(self):
        return self.num_frame

    def close(self):
        self.cap.release()

 
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
