from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import *
import pyqtgraph as pg
import time

class TimeAxisItem(pg.AxisItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setLabel(text='Real Time', units=None)
        self.enableAutoSIPrefix(False)

    def tickStrings(self, values, scale, spacing):
        """ override 하여, tick 옆에 써지는 문자를 원하는대로 수정함.
            values --> x축 값들   ; 숫자로 이루어진 Itarable data --> ex) List[int]
        """
        return [time.strftime("%H:%M:%S", time.localtime(local_time)) for local_time in values]


class Realime_graph(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.pw1 = pg.PlotWidget(
            title="The number of people",
            labels={'left': 'People-Counting'},
            axisItems={'bottom': TimeAxisItem(orientation='bottom')}
        )

        self.pw1.setTitle("The number of people", **{'color': 'w', 'size': '14pt'})

        self.pw2 = pg.PlotWidget(
            title="The risk by distance",
            labels={'left': 'Distance-Risk-Score'},
            axisItems={'bottom': TimeAxisItem(orientation='bottom')}
        )
        self.pw2.setTitle("The risk by distance", **{'color': 'w', 'size': '14pt'})

        self.pw3 = pg.PlotWidget(
            title="The dangers of fighting action",
            labels={'left': 'Fight-Dangerous-score'},
            axisItems={'bottom': TimeAxisItem(orientation='bottom')}
        )
        self.pw3.setTitle("The dangers of fighting action", **{'color': 'w', 'size': '14pt'})

        vbox1 = QVBoxLayout()
        hbox1 = QHBoxLayout()
        hbox2 = QHBoxLayout()

        hbox1.addWidget(self.pw1)
        hbox1.addWidget(self.pw2)
        hbox2.addWidget(self.pw3)

        vbox1.addLayout(hbox1)
        vbox1.addLayout(hbox2)

        self.setLayout(vbox1)

        self.pw1.setYRange(-1, 5, padding=0)
        self.pw2.setYRange(-1, 5, padding=0)
        self.pw3.setYRange(-1, 100, padding=0)

        time_data = int(time.time())

        self.pw1.setXRange(time_data - 10, time_data + 1)  # 생략 가능.
        self.pw2.setXRange(time_data - 10, time_data + 1)  # 생략 가능.
        self.pw3.setXRange(time_data - 10, time_data + 1)  # 생략 가능.

        self.pw1.showGrid(x=True, y=True)
        self.pw2.showGrid(x=True, y=True)
        self.pw3.showGrid(x=True, y=True)

        # (29, 185, 84)연두 / (102, 252, 241) 하늘 / (200, 200, 255) 보라
        self.plotData1 = {'x': [], 'y': []}
        self.plotData2_1 = {'x': [], 'y': []}
        self.plotData2_2 = {'x': [], 'y': []}
        self.plotData3 = {'x': [], 'y': []}

        self.pdi1 = self.pw1.plot(pen=pg.mkPen((000, 255, 255), width=6))  # PlotDataItem obj 반환.

        self.pdi2_1 = self.pw2.plot(pen=pg.mkPen((255, 255, 62), width=6))
        self.pdi2_2 = self.pw2.plot(pen=pg.mkPen((238, 130, 238), width=6))

        self.pdi3 = self.pw3.plot(pen=pg.mkPen(255, 100, 100, width=7), fillLevel=-0.9, brush=(255, 0, 0, 50))

    def update_plot(self, x=0, y1=100, y2=100,y2_2=100, y3=100):
        self.plotData1['x'].append(x)
        self.plotData1['y'].append(y1)

        self.plotData2_1['x'].append(x)
        self.plotData2_1['y'].append(y2)

        self.plotData2_2['x'].append(x)
        self.plotData2_2['y'].append(y2_2)

        self.plotData3['x'].append(x)
        self.plotData3['y'].append(y3)

        # 항상 x축 시간을 최근 범위만 보여줌.
        self.pw1.setXRange(x - 10, x + 1, padding=0)
        self.pw2.setXRange(x - 10, x + 1, padding=0)
        self.pw3.setXRange(x - 10, x + 1, padding=0)

        self.pdi1.setData(self.plotData1['x'], self.plotData1['y'], symbol='o', symbolpen='r', symbolSize=14, symbolBrush=(000, 200, 200, 255))
        self.pdi2_1.setData(self.plotData2_1['x'], self.plotData2_1['y'])
        self.pdi2_2.setData(self.plotData2_2['x'], self.plotData2_2['y'])
        self.pdi3.setData(self.plotData3['x'], self.plotData3['y'])


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    ex = Realime_graph()

    def get_data():
        new_time_data = int(time.time())
        #ex.update_plot(new_time_data, datetime.datetime.now().second*40)
        ex.update_plot(new_time_data, 100,200)

    mytimer = QTimer()
    mytimer.start(1000)  # 1초마다 갱신 위함...
    mytimer.timeout.connect(get_data)

    ex.show()
    sys.exit(app.exec_())