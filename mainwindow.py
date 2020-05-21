# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.14.2
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox
import numpy as np
import cv2
from PIL import Image
import nibabel as nib
import math
from segmentation import segment_img,lv_localize,segment_img_with_center

class Ui_MainWindow(object):

    file_path = ''
    slice_data = None
    total_slice = 0
    current_slice = 100
    current_img = 0
    total_img = 0
    current_seg_img = 0
    total_seg_img = 0
    header = None
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 700)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.loadFileButton = QtWidgets.QPushButton(self.centralwidget)
        self.loadFileButton.setGeometry(QtCore.QRect(30, 20, 140, 25))
        self.loadFileButton.setObjectName("loadFileButton")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(30, 70, 150, 17))
        self.label.setObjectName("label")
        self.inputSliceNum = QtWidgets.QLineEdit(self.centralwidget)
        self.inputSliceNum.setGeometry(QtCore.QRect(190, 70, 31, 21))
        self.inputSliceNum.setObjectName("inputSliceNum")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(230, 70, 16, 17))
        self.label_2.setObjectName("label_2")
        self.totalSlice = QtWidgets.QLabel(self.centralwidget)
        self.totalSlice.setGeometry(QtCore.QRect(240, 70, 31, 17))
        self.totalSlice.setText("")
        self.totalSlice.setObjectName("totalSlice")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(120, 120, 150, 17))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(560, 120, 150, 17))
        self.label_4.setObjectName("label_4")
        self.imageLabel = QtWidgets.QLabel(self.centralwidget)
        self.imageLabel.setGeometry(QtCore.QRect(20, 150, 300, 300))
        self.imageLabel.setText("")
        self.imageLabel.setObjectName("imageLabel")
        self.segImageLabel = QtWidgets.QLabel(self.centralwidget)
        self.segImageLabel.setGeometry(QtCore.QRect(470, 150, 300, 300))
        self.segImageLabel.setText("")
        self.segImageLabel.setObjectName("segImageLabel")
        self.runButton = QtWidgets.QPushButton(self.centralwidget)
        self.runButton.setGeometry(QtCore.QRect(350, 360, 89, 25))
        self.runButton.setObjectName("runButton")
        self.horizontalScrollBar = QtWidgets.QScrollBar(self.centralwidget)
        self.horizontalScrollBar.setGeometry(QtCore.QRect(90, 460, 160, 16))
        self.horizontalScrollBar.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalScrollBar.setObjectName("horizontalScrollBar")
        self.currentImg = QtWidgets.QLabel(self.centralwidget)
        self.currentImg.setGeometry(QtCore.QRect(140, 490, 16, 17))
        self.currentImg.setObjectName("currentImg")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(160, 490, 16, 17))
        self.label_8.setObjectName("label_8")
        self.totalImg = QtWidgets.QLabel(self.centralwidget)
        self.totalImg.setGeometry(QtCore.QRect(180, 490, 21, 17))
        self.totalImg.setText("")
        self.totalImg.setObjectName("totalImg")
        self.filePathLabel = QtWidgets.QLineEdit(self.centralwidget)
        self.filePathLabel.setGeometry(QtCore.QRect(200, 20, 460, 20))
        self.filePathLabel.setStyleSheet("background-color:\"transparent\"")
        self.filePathLabel.setFrame(False)
        self.filePathLabel.setReadOnly(True)
        self.filePathLabel.setObjectName("filePathLabel")
        self.calculateButton = QtWidgets.QPushButton(self.centralwidget)
        self.calculateButton.setGeometry(QtCore.QRect(350, 570, 89, 25))
        self.calculateButton.setObjectName("calculateButton")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(150, 540, 120, 17))
        self.label_5.setObjectName("label_5")
        self.inputEDNum = QtWidgets.QLineEdit(self.centralwidget)
        self.inputEDNum.setGeometry(QtCore.QRect(280, 540, 31, 21))
        self.inputEDNum.setObjectName("inputEDNum")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(150, 600, 120, 17))
        self.label_6.setObjectName("label_6")
        self.inputESNum = QtWidgets.QLineEdit(self.centralwidget)
        self.inputESNum.setGeometry(QtCore.QRect(280, 600, 31, 21))
        self.inputESNum.setObjectName("inputESNum")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(480, 540, 81, 17))
        self.label_7.setObjectName("label_7")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(480, 570, 81, 17))
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(480, 600, 131, 17))
        self.label_10.setObjectName("label_10")
        self.edVolume = QtWidgets.QLabel(self.centralwidget)
        self.edVolume.setGeometry(QtCore.QRect(630, 540, 67, 17))
        self.edVolume.setText("")
        self.edVolume.setObjectName("edVolume")
        self.esVolume = QtWidgets.QLabel(self.centralwidget)
        self.esVolume.setGeometry(QtCore.QRect(630, 570, 67, 17))
        self.esVolume.setText("")
        self.esVolume.setObjectName("esVolume")
        self.ejection_fraction = QtWidgets.QLabel(self.centralwidget)
        self.ejection_fraction.setGeometry(QtCore.QRect(630, 600, 67, 17))
        self.ejection_fraction.setText("")
        self.ejection_fraction.setObjectName("ejection_fraction")
        self.label_11 = QtWidgets.QLabel(self.centralwidget)
        self.label_11.setGeometry(QtCore.QRect(710, 540, 67, 17))
        self.label_11.setObjectName("label_11")
        self.label_12 = QtWidgets.QLabel(self.centralwidget)
        self.label_12.setGeometry(QtCore.QRect(710, 570, 67, 17))
        self.label_12.setObjectName("label_12")
        self.label_13 = QtWidgets.QLabel(self.centralwidget)
        self.label_13.setGeometry(QtCore.QRect(710, 600, 67, 17))
        self.label_13.setObjectName("label_13")
        self.label_14 = QtWidgets.QLabel(self.centralwidget)
        self.label_14.setGeometry(QtCore.QRect(346, 170, 101, 16))
        self.label_14.setText("")
        self.label_14.setObjectName("label_14")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.filePathLabel.textChanged.connect(self.getNiftiImage)
        self.inputSliceNum.textEdited.connect(self.onCurrentSliceChanged)
        self.horizontalScrollBar.valueChanged.connect(self.onScrollBarChanged)
        self.runButton.clicked.connect(self.onRunButtonPressed)
        self.calculateButton.clicked.connect(self.onCalculateButtonPressed)
        self.loadFileButton.clicked.connect(self.open_file)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.loadFileButton.setText(_translate("MainWindow", "Load Nifti Image"))
        self.label.setText(_translate("MainWindow", "Current slice number :"))
        self.label_2.setText(_translate("MainWindow", "/"))
        self.label_3.setText(_translate("MainWindow", "Original Images"))
        self.label_4.setText(_translate("MainWindow", "Segmented Images"))
        self.runButton.setText(_translate("MainWindow", "Run"))
        self.currentImg.setText(_translate("MainWindow", "1"))
        self.label_8.setText(_translate("MainWindow", "/"))
        self.calculateButton.setText(_translate("MainWindow", "Calculate"))
        self.label_5.setText(_translate("MainWindow", "ED Slice number :"))
        self.label_6.setText(_translate("MainWindow", "ES Slice number :"))
        self.label_7.setText(_translate("MainWindow", "ED Volume : "))
        self.label_9.setText(_translate("MainWindow", "ES Volume :"))
        self.label_10.setText(_translate("MainWindow", "Ejection Fraction : "))
        self.label_11.setText(_translate("MainWindow", "mm3"))
        self.label_12.setText(_translate("MainWindow", "mm3"))
        self.label_13.setText(_translate("MainWindow", "%"))


    def getNiftiImage(self, file_path):
        
        img = nib.load(file_path) 
        # self.setFilePathLabel(fname)
        self.header = img.header.get_zooms()
        self.slice_data = img.get_fdata()
        self.setTotalSlice(self.slice_data.shape[3]) 
        self.setCurrentSlice(0)
        self.inputSliceNum.setText(str(self.current_slice+1))
        self.setTotalImage(self.slice_data[:,:,:,self.current_slice].shape[2])
        self.setCurrentImage(0)
        self.setImage(self.current_img)
        self.horizontalScrollBar.setMaximum(self.total_img-1)
        self.horizontalScrollBar.setValue(0)
        self.segImageLabel.clear()
        self.inputESNum.clear()
        self.inputEDNum.clear()
        self.edVolume.clear()
        self.esVolume.clear()
        self.ejection_fraction.clear()

    def setImage(self, current_img):

        # ed_img = nib.load("../training/patient001/patient001_4d.nii.gz")
        # ed_img_data = self.ed_img.get_fdata()
        if 0 <= current_img < self.total_img:
            max_pixel_value = self.slice_data.max()

            if max_pixel_value > 0:
                multiplier = 255.0 / max_pixel_value
            else:
                multiplier = 1.0

            image = self.slice_data[:,:,current_img,self.current_slice]
            image = np.rot90(image[:,::-1],1) * multiplier
            image_data = cv2.convertScaleAbs(image)
            height, width = image_data.shape 
            qImg = QtGui.QImage(image_data.data,width,height,QtGui.QImage.Format_Indexed8)
            pix = QtGui.QPixmap(qImg)
            self.imageLabel.setPixmap(pix)
            self.imageLabel.setAlignment(QtCore.Qt.AlignCenter)
        else:
            print('Invalid image')

    def setTotalSlice(self, total_slice):

        if total_slice:
            self.total_slice = total_slice
            self.totalSlice.setText(str(self.total_slice))
        else:
            print("Total Slice is not valid.")

    def setCurrentSlice(self, current_slice):
        # if self.current_slice == 0:
        #     self.current_slice = 1
        #     self.inputSliceNum.setText(str(self.current_slice)
        #     return            

        if 0 <= current_slice < self.total_slice:
            self.current_slice = current_slice
        else:
            print("Invalid Current Slice number")

    def setFilePathLabel(self, file_path):
        if file_path:
            self.file_path = file_path
            self.filePathLabel.setText(self.file_path)
        else:
            print('Invalid file path')

    def setTotalImage(self, total_img):
        if total_img:
            self.total_img = total_img
            self.totalImg.setText(str(self.total_img))
        else:
            print('Invalid total image')

    def setCurrentImage(self, current_img):
        if 0 <= current_img < self.total_img:
            self.current_img = current_img
            self.currentImg.setText(str(self.current_img+1))
        else:
            print('Invalid current image')

    def setTotalSegImage(self, total_seg_img):
        if total_seg_img:
            self.total_seg_img = total_seg_img
            self.totalSegImg.setText(str(self.total_seg_img))
        else:
            print('Invalid total seg image')

    def setCurrentSegImage(self, current_seg_img):
        if 0 <= current_seg_img < self.total_seg_img:
            self.current_seg_img = current_seg_img
            self.currentSegImg.setText(str(self.current_seg_img+1))
        else:
            print('Invalid current seg image')

    def onCurrentSliceChanged(self, str_current_slice):
        if not str_current_slice:
            return
        if not int(str_current_slice):
            return
        current_slice = int(str_current_slice)-1
        if current_slice != self.current_slice:
            self.setCurrentSlice(current_slice)
            self.setTotalImage(self.slice_data[:,:,:,self.current_slice].shape[2])
            self.setCurrentImage(0)
            self.setImage(self.current_img)
            self.horizontalScrollBar.setValue(0)

    def onScrollBarChanged(self, str_value):
        
        int_value = int(str_value)
        self.current_img = int_value
        self.currentImg.setText(str(self.current_img+1))
        self.setImage(self.current_img)

    def onRunButtonPressed(self):
        self.runButton.setEnabled(False)
        self.inputSliceNum.setEnabled(False)
        self.horizontalScrollBar.setEnabled(False)
        self.inputEDNum.setEnabled(False)
        self.inputESNum.setEnabled(False)
        self.calculateButton.setEnabled(False)
        max_pixel_value = self.slice_data.max()

        if max_pixel_value > 0:
            multiplier = 255.0 / max_pixel_value
        else:
            multiplier = 1.0
        ed_mid_slice = int(self.slice_data[:,:,:,self.current_slice].shape[2]/2)
        ed_mid_image = self.slice_data[:,:,ed_mid_slice,self.current_slice]
        ed_mid_image = np.rot90(ed_mid_image[:,::-1],1)*multiplier

        ed_current_image = self.slice_data[:,:,self.current_img,self.current_slice]
        ed_current_image = np.rot90(ed_current_image[:,::-1],1)*multiplier
        lv_center_x,lv_center_y = lv_localize(ed_mid_image)
        segmentImg, area = segment_img_with_center(ed_current_image,lv_center_x,lv_center_y)
        if segmentImg.any():
            height, width, channel = segmentImg.shape 
            bytesPerLine = 3 * width
            qImg = QtGui.QImage(segmentImg.data,width,height,bytesPerLine,QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap(qImg)
            self.segImageLabel.setPixmap(pix)
            self.segImageLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.runButton.setEnabled(True)
        self.inputSliceNum.setEnabled(True)
        self.horizontalScrollBar.setEnabled(True)
        self.inputEDNum.setEnabled(True)
        self.inputESNum.setEnabled(True)
        self.calculateButton.setEnabled(True)

    def onCalculateButtonPressed(self):
        area_ed = 0
        area_es = 0
        volume_ed = 0
        volume_es = 0
        ejection_fraction = 0
        if not self.inputEDNum.text():
            msg = QMessageBox.critical(None,'Error','ED and ES slice number should not be empty!',QMessageBox.Ok)
            return
        if not self.inputESNum.text():
            msg = QMessageBox.critical(None,'Error','ED and ES slice number should not be empty!',QMessageBox.Ok)
            return
        ed_num = int(self.inputEDNum.text())-1
        es_num = int(self.inputESNum.text())-1
        total_ed = self.slice_data[:,:,:,ed_num].shape[2]
        total_es = self.slice_data[:,:,:,es_num].shape[2]
        ed_mid_slice = int(total_ed/2)
        self.runButton.setEnabled(False)
        self.inputSliceNum.setEnabled(False)
        self.horizontalScrollBar.setEnabled(False)
        self.inputEDNum.setEnabled(False)
        self.inputESNum.setEnabled(False)
        self.calculateButton.setEnabled(False)
        lv_center_x,lv_center_y = lv_localize(self.slice_data[:,:,ed_mid_slice,ed_num])
        for i in range(total_ed):
            segmentImg, area1 = segment_img_with_center(self.slice_data[:,:,i,ed_num],lv_center_x,lv_center_y)
            area_ed = area_ed + area1
        lv_center_x,lv_center_y = lv_localize(self.slice_data[:,:,ed_mid_slice,es_num])
        for j in range(total_es):
            segmentImg, area1 = segment_img_with_center(self.slice_data[:,:,j,es_num],lv_center_x,lv_center_y)
            area_es = area_es + area1
        volume_ed = area_ed * self.header[0] * self.header[1] * self.header[2] /1000
        volume_es = area_es * self.header[0] * self.header[1] * self.header[2] /1000
        volume_ed_int = int(volume_ed*10000) / 10000
        volume_es_int = int(volume_es*10000) / 10000
        ejection_fraction = ((volume_ed - volume_es)/volume_ed)*100
        ejection_fraction_int = int(ejection_fraction*10000)/10000
        self.edVolume.setText(str(volume_ed_int))
        self.esVolume.setText(str(volume_es_int))
        self.ejection_fraction.setText(str(ejection_fraction_int))

        self.runButton.setEnabled(True)
        self.inputSliceNum.setEnabled(True)
        self.horizontalScrollBar.setEnabled(True)
        self.inputEDNum.setEnabled(True)
        self.inputESNum.setEnabled(True)
        self.calculateButton.setEnabled(True)

    def open_file(self):
        fname, _filter = QtWidgets.QFileDialog.getOpenFileName(None, "Open File", '.', "(*.nii.gz)")
        if fname:
            self.file_path = fname
            self.setFilePathLabel(fname)

    def showAlert(self):
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Critical)
        msg.setText("ED and ES slice number should not be empty!")
        msg.setWindowTitle("Error")
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
