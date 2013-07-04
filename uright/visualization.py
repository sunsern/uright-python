import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


def draw_ink(ax, ink, node_size=0.1, 
             penup_color='#0000FF',
             pendown_color='#00FF00',
             node_colors=None,
             show_order=False):
    current_stroke_id = 0
    for i in range(ink.shape[0]):
        # skip pen-up
        if ink[i,4] > 0:
            continue
            
        e = Circle(xy=(ink[i,0],ink[i,1]),radius=node_size, alpha=0.5)
        ax.add_artist(e)
        
        # pen-up/pen-down
        if i == 0 or ink[i-1,4] > 0:
            e.set_color(pendown_color)
            e.set_linewidth(5.0)
            current_stroke_id += 1

            if show_order:
                ax.text(ink[i,0]+node_size,ink[i,1],"[%d]"%current_stroke_id)

        elif i == ink.shape[0]-1 or ink[i+1,4] > 0:
            e.set_color(penup_color)
            e.set_linewidth(5.0)
            
        if node_colors is not None:
            e.set_facecolor(node_colors[i])
        
        # draw arrow
        if (i < ink.shape[0]-1 and
            ink[i,4] < 1 and
            ink[i+1,4] < 1):
            dx = ink[i+1,0]-ink[i,0]
            dy = ink[i+1,1]-ink[i,1]
            z = np.sqrt(dx*dx+dy*dy)
            dx = dx / max(z,1e-5)
            dy = dy / max(z,1e-5)
            ax.arrow(ink[i,0]-0.5*node_size*dx,
                     ink[i,1]-0.5*node_size*dy,
                     node_size*dx, node_size*dy,
                     fc="k", ec="k", alpha=0.5, width=0.007,
                     head_width=0.03, head_length=0.02,
                     length_includes_head=True)            

    # adjust plot size
    ax.axis('equal')
    #ax.axis('scaled')
    #ax.set_xlim(np.nanmin(ink[:,0])-0.2,
    #            np.nanmax(ink[:,0])+0.2)
    #ax.set_ylim(np.nanmax(ink[:,1])+0.2,
    #            np.nanmin(ink[:,1])-0.2)
    #ax.set_xticks([])
    #ax.set_yticks([])
