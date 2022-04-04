else:

app = QtWidgets.QApplication(sys.argv)
win = QtWidgets.QWidget()
vbox = QtWidgets.QVBoxLayout()
vbox2 = QtWidgets.QHBoxLayout()
VideoSignal1 = QtWidgets.QLabel()
combo_start = QComboBox()
btn_start = QtWidgets.QPushButton("카메라 켜기")
btn_stop = QtWidgets.QPushButton("카메라 끄기")
win.setWindowTitle("감시카메라")
win.resize(500, 200)

combo_start.addItem("배회영역 설정")
check = QtWidgets.QPushButton("선택")

btn_start.clicked.connect(start)
btn_stop.clicked.connect(그만)


def connecttion():
    if combo_start.currentText() == "배회영역 설정":
        print(combo_start.currentText())
        global redrectangle_roi_pyqt
        print("start roi..")
        redrectangle_roi_pyqt = True


check.clicked.connect(connecttion)
vbox.addWidget(VideoSignal1)
vbox.addLayout(vbox2)

vbox2.addWidget(btn_start)
vbox2.addWidget(btn_stop)
vbox.addWidget(combo_start)
vbox.addWidget(check)
win.setLayout(vbox)
win.show()
sys.exit(app.exec_())