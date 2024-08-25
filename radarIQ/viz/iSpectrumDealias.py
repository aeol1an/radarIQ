import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mplPath

LEFT = 1
RIGHT = 3

#code from https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
def onSegment(p, q, r): 
    if ( (q[0] <= max(p[0], r[0])) and (q[0] >= min(p[0], r[0])) and 
           (q[1] <= max(p[1], r[1])) and (q[1] >= min(p[1], r[1]))): 
        return True
    return False
  
def orientation(p, q, r): 
    val = (float(q[1] - p[1]) * (r[0] - q[0])) - (float(q[0] - p[0]) * (r[1] - q[1])) 
    if (val > 0): 
        return 1
    elif (val < 0): 
        return 2
    else: 
        return 0

def doIntersect(p1,q1,p2,q2): 
    o1 = orientation(p1, q1, p2) 
    o2 = orientation(p1, q1, q2) 
    o3 = orientation(p2, q2, p1) 
    o4 = orientation(p2, q2, q1) 
  
    if ((o1 != o2) and (o3 != o4)): 
        return True
    if ((o1 == 0) and onSegment(p1, p2, q1)): 
        return True
    if ((o2 == 0) and onSegment(p1, q2, q1)): 
        return True
    if ((o3 == 0) and onSegment(p2, p1, q2)): 
        return True
    if ((o4 == 0) and onSegment(p2, q1, q2)): 
        return True

    return False
#end code from geeksforgeeks

def samePoint(p, q):
    if p[0] == q[0] and p[1] == q[1]:
        return True
    return False

def bound(val, upper, lower):
    if val > upper:
        return upper
    elif val < lower:
        return lower
    return val

def dealiasOnce(fields):
    field = fields[0]
    ny, nx = field.shape
    
    xx, yy = np.meshgrid(range(nx), range(ny))
    
    fig = plt.figure(figsize=[8, 6])
    plt.pcolormesh(xx, yy, field)
    
    selectionDone = False
    points = []
    selectionResult = np.array([])
    
    def addPoint(event):
        nonlocal selectionDone
        nonlocal points
        nonlocal selectionResult
        
        if (not event.dblclick) or (event.button != LEFT and event.button != RIGHT):
            return
        
        x = round(event.xdata)
        x = bound(x, nx-1, 0)
        y = round(event.ydata)
        y = bound(y, ny-1, 0)
        point = [x, y]
        
        #error finding to see if cycle or something like that
        if (not selectionDone) and (len(points) >= 3) and (not samePoint(points[0], point)) and (event.button != RIGHT):
            for i in range(len(points)-2):
                p1 = points[i]
                q1 = points[i+1]
                p2 = points[-1]
                if doIntersect(p1, q1, p2, point):
                    print("Intersecting points detected. Try again.")
                    return
                
        #logic to add to points array
        if not selectionDone:
            if event.button == RIGHT and (len(points) >= 3):
                point = points[0]
            points.append(point)
            plt.scatter(point[0], point[1], color='r')
            if len(points) >= 2:
                npPoints = np.array(points)
                npPoints = npPoints[-2:,].T
                plt.plot(npPoints[0], npPoints[1], color='r')
            event.canvas.draw()
            event.canvas.flush_events()

        #if made a circle or first and last set done to true
        if samePoint(points[0], points[-1]) and (len(points) >= 3):
            selectionDone = True
        
        #if done is true, calculate numpy boolean array with things to move over after selecting side of line
        if not selectionDone:
            return
        path = mplPath.Path(points)
        allpoints = np.array([xx,yy]).transpose(1,2,0).reshape(-1,2)
        selectionResultPos = path.contains_points(allpoints, radius=1)
        selectionResultNeg = path.contains_points(allpoints, radius=-1)
        if np.sum(selectionResultPos) > np.sum(selectionResultNeg):
            selectionResult = selectionResultPos
        else:
            selectionResult = selectionResultNeg
        selectionResult = selectionResult.reshape(ny,nx)
        fig.canvas.mpl_disconnect(boundary_handler)
        
    boundary_handler = fig.canvas.mpl_connect('button_press_event', addPoint)
    
    
    unfoldDone = False
    unfoldDirection = ""
    unfoldsAdded = np.array([0,0])
    
    def performUnfold(event):
        nonlocal unfoldDone
        nonlocal unfoldDirection
        nonlocal unfoldsAdded
        
        if not selectionDone:
            return
        if not (event.key in ['left', 'right', 'enter']):
            return
        
        if event.key == 'enter':
            plt.close()
            return
        
        if unfoldDone:
            return
        isoSelection = np.array(field)
        isoSelection[~selectionResult] = np.nan
        fieldNan = np.array(field)
        fieldNan[selectionResult] = np.nan
        
        if event.key == 'left':
            unfoldDirection = 'left'
            unfoldsAdded += np.array([1,0])
            newField = np.concatenate((isoSelection, fieldNan), axis=1)
        elif event.key == 'right':
            unfoldDirection = 'right'
            unfoldsAdded += np.array([0,1])
            newField = np.concatenate((fieldNan, isoSelection), axis=1)
            
        plt.clf()
        newxx, newyy = np.meshgrid(range(newField.shape[1]), range(newField.shape[0]))
        plt.pcolormesh(newxx, newyy, newField)
        event.canvas.draw()
        event.canvas.flush_events()
    
    fig.canvas.mpl_connect('key_press_event', performUnfold)
    
    plt.show(block=True)
    
    plt.rcParams['keymap.back'].append('left')
    plt.rcParams['keymap.forward'].append('right')
    
    retFields = []
    for field in fields:
        isoSelection = np.array(field)
        isoSelection[~selectionResult] = np.nan
        fieldNan = np.array(field)
        fieldNan[selectionResult] = np.nan
        
        if unfoldDirection == 'left':
            newField = np.concatenate((isoSelection, fieldNan), axis=1)
        elif unfoldDirection == 'left':
            newField = np.concatenate((fieldNan, isoSelection), axis=1)
            
        retFields.append(newField)
        
    return retFields, unfoldsAdded