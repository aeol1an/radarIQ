import radarIQ
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use(backend="TkAgg")
from matplotlib import pyplot as plt
import time

csvPath = Path().cwd()/"cases"/"may232024"/"goodData.csv"
if not csvPath.exists():
    with open(csvPath, "w") as f:
        f.write("fileName,good,startPulse,endPulse\n")

for i in range(50, 175):
    data = radarIQ.raxpolrkc(root=Path().cwd(), caseName='may232024', fileNo=i)
    pulseBounds = []
    def onclick(event):
        if event.dblclick:
            xclick = event.xdata
            if xclick < 0:
                xclick = 0
            elif xclick > data.pulses.shape[0] - 1:
                xclick = data.pulses.shape[0] - 1
            else:
                xclick = round(xclick/100) * 100
            pulseBounds.append(xclick)
            event.inaxes.axvline(x=xclick, color='r')
            event.canvas.draw()
            event.canvas.flush_events()
            if len(pulseBounds) == 2:
                time.sleep(1)
                plt.close()
                pulseBounds.sort()
                goodBool = not (pulseBounds[0] == 0 and pulseBounds[1] == 0)
                with open(csvPath, "a") as f:
                    f.write(f"{Path(data.filename).name},{goodBool},"
                            f"{pulseBounds[0]},{pulseBounds[1]}\n")

    plt.plot(np.arange(0, len(data.azArray())), data.azArray())
    plt.ylim([-100, 460])

    cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick)
    plt.show()