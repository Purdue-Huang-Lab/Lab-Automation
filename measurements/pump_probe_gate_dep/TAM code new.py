"""
Author: Jonas M. Peterson
"""
from matplotlib.widgets import Slider, Button, TextBox,RadioButtons,CheckButtons
#pip install openpyxl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.ndimage import uniform_filter
import os
import glob

from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.integrate import quad
from scipy.interpolate import interp1d
import time
def format_ticks(value, pos):
    return str(int(value))



def gaussian_with_offset(x, amplitude, mean, stddev, offset):
    return amplitude * np.exp(-((x - mean) / (2 * stddev)) ** 2) + offset



def sincplusgauss(x, A, o, w, c, B, J):
    sinc_part = np.where(np.abs(x - o) < 1e-5, w, np.sin(w * (x - o)) / ((x - o)))
    #sinc_part = np.sinc(w*(x-o))
    #sinc_part = jve(0,w * (x - o))
    normalization_factor = 1 / (J * np.sqrt(2 * np.pi))
    exponent = -0.5 * ((x - o) / J) ** 2
    gaussian_part = B * normalization_factor * np.exp(exponent)
    #gaussian_part =B  * np.sinc(J*(x-o))
    #A * sinc_part + gaussian_part + c
    return A * sinc_part + gaussian_part + c
def integrand_sincplusgauss_squared(x, A, o, w, c, B, J):
    return  sincplusgauss(x, A, o, w, c, B, J)**2
def integrand_Gauss(x, B, J):
    return  (B * np.exp(-((x) / (2 * J)) ** 2)) **2


def calculate_rms_error(x_values, params, errors):
    A, o, w, c, B, J = params

    # Partial derivatives of RMS with respect to each parameter
    dRMS_dA = 0.5 * np.sqrt(np.sum((sincplusgauss(x_values, *params)) ** 2)) / len(x_values)
    #print("---")
    #print(dRMS_dA)
    dRMS_do = 0.5 * A * np.sum(np.cos(w * (x_values - o)) / ((x_values - o) ** 2)) / len(x_values)
    #print(dRMS_do)
    dRMS_do = 0
    dRMS_dw = 0.5 * A * np.sum((x_values - o) * np.cos(w * (x_values - o)) / ((x_values - o) ** 2)) / len(x_values)
    #print(dRMS_dw)
    dRMS_dc = 0.5 * np.sum(np.ones_like(x_values)) / len(x_values)
    #print(dRMS_dc)
    dRMS_dB = 0.5 * np.sqrt(np.sum(np.exp(-((x_values - o) / J) ** 2))) / len(x_values)
    #print(dRMS_dB)
    dRMS_dJ = 0.5 * B * np.sqrt(np.pi / 2) * np.sum(
        ((x_values - o) ** 2) / J ** 3 * np.exp(-((x_values - o) / J) ** 2)) / len(x_values)
    #print(dRMS_dJ)
    #print("---")

    # Error propagation formula
    rms_error = np.sqrt((dRMS_dA * errors[0]) ** 2 + (dRMS_do * errors[1]) ** 2 + (dRMS_dw * errors[2]) ** 2
                        + (dRMS_dc * errors[3]) ** 2 + (dRMS_dB * errors[4]) ** 2 + (dRMS_dJ * errors[5]) ** 2)

    return rms_error

def create_gradient(start_color, end_color, num_steps):

    r1, g1, b1 = start_color
    r2, g2, b2 = end_color

    gradient = []
    for i in range(num_steps):
        ratio = i / (num_steps - 1)
        r = int((1 - ratio) * r1 + ratio * r2)
        g = int((1 - ratio) * g1 + ratio * g2)
        b = int((1 - ratio) * b1 + ratio * b2)
        gradient.append((r, g, b))

    return gradient




def convert(inp, m, b):
    return m * inp + b


def convert2(inp, m, b):
    return m * (inp + b)
def find_closest_index_in_range(array, value, lower_limit, upper_limit):
    # Filter the array within the specified range
    filtered_array = array[(array >= lower_limit) & (array <= upper_limit)]

    # Find the index with the minimum absolute difference in the filtered array
    closest_index_in_filtered = np.argmin(np.abs(filtered_array - value))

    # Get the index of the closest value in the original array
    closest_index = np.where(array == filtered_array[closest_index_in_filtered])[0][0]

    return closest_index


def find_closest_index(array, value):
    # Calculate the absolute differences between each element and the target value
    absolute_diff = np.abs(array - value)

    # Find the index with the minimum absolute difference
    closest_index = np.argmin(absolute_diff)

    return closest_index


'''
CHANGE THINGS IN HERE FOR YOUR SPECIFIC STUFF
'''
# filename and param file name
# If you give only the document name, it will be resolved against BASE_DIR.
# Use the directory of this script as BASE_DIR (fallback to CWD if __file__ is unavailable).
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()
def resolve_path(name: str) -> str:
    # If name already contains a directory or is absolute, return as-is
    if os.path.isabs(name) or os.path.dirname(name):
        return name
    return os.path.join(BASE_DIR, name)

# You can set these to just the file names, or full paths if you prefer
filename = resolve_path("4.2nww 1 ND 1sum.txt")
filename2 = resolve_path("15nww 1 ND 1para.txt")
#wavelength cab
m = 0.44
b  =630
# sizeofimage
N1 = 128  # number of rows in raw image
N2 = 480  # number of columns in raw image
crop1 = 0  # amount of pixels cut off the from of the left of image (on the long axis)
crop2 = 0 # amount of pixels cut off the from of the right image (on the long axis)
crop3 = 0 # amount of pixels cut off the top of the top of image (max is N1-1)
crop4 = 128 # amount of pixels included from the top (max is N1)

# calibration for image wavelength




# calibrate for image size ( pixels per micrometer)
pixelperum = 0.08

# time zero if you want for quick export
TZERO  = 91.1
 # end time for window for quick export
ETIME  = 300
#number of points in quick export
numberofplots = 10

#parameters from chirp correction
# enter like this Coeff = np.array([A,B,C])
Coeff = np.array([-2.09211940e-04,3.02271550e-01,-1.03070449e+02])

#smoothkernelsize
smthk = 6





# file = open(filename + ".csv")
file = open(filename)
# file = open("worku 1 w 1 ND 1sum.txt")
# images = np.loadtxt(file, delimiter=",")
images = pd.read_csv(filename, delimiter='\t', header=None)
#images = pd.read_csv(filename, header=None)

images[images == '--'] = np.nan
images = images.values
#images = images.astype(values)
#images[images == '--'] = np.nan
#images = images.astype(float)

filepams = open(filename2)
# Time = np.loadtxt(filename3, delimiter=",")
Time2 = pd.read_csv(filename2, delimiter='\t')
Time2.columns = Time2.columns.astype(float)
# print(Time2.columns)
arrayoftime = np.array(Time2.columns)

RTIME = arrayoftime
Itime = np.array(range(0, len(RTIME), 1))
Time = Itime

ITZERO = find_closest_index(arrayoftime,TZERO)

IETIME = find_closest_index(arrayoftime,ETIME)
recon = {}

for i in range(0, len(Time)):
    leng = i * N2
    width = N1
    raw = np.rot90(images[leng + crop1:leng + N2 - crop2, N1 - crop4:N1 - crop3])
    # Faster box smoothing using uniform_filter
    smoothed_array = uniform_filter(raw.astype(np.float32), size=smthk, mode='nearest')
    recon[Time[i]] = smoothed_array

plotmax = 2  # max of colorplot

plotmin = -1  # min of color plot

# fig, (ax1, ax2, ax3,ax4) = plt.subplots(2,2, figsize=(15, 5))
#fig, ((ax1, ax2), (ax3, ax4),(ax5, ax6)) = plt.subplots(3, 2, figsize=(10, 8))

# plt.tight_layout()
#plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4)


slope = m
interce = b
pix = crop1
pixvec = np.arange(recon[2].shape[1])
finalpix = pixvec + pix
newp = np.arange(recon[2].shape[0])
subt = -1 * int(recon[2].shape[0]) / 2
newpos = convert2(newp, pixelperum, subt)
newwave = convert(finalpix, slope, interce)

'''
CHIRP CORRECTION
'''
# Skipped for performance (compute but do not use). If needed, re-enable.
# coefficients = Coeff
# truetzero = np.polyval(coefficients, newwave)
# truetzeroind = np.zeros(len(truetzero))
# for i in range(0, len(truetzero)):
#     truetzeroind[i] = find_closest_index(arrayoftime, float(truetzero[i]))
# arrayofTIME = np.zeros((len(Time), len(newwave)))
# for i in range(0, len(newwave)):
#      array1 = arrayoftime - truetzero[i]
#      arrayofTIME[:, i] = array1
# univmin = np.max(arrayofTIME[0, :])
# univmax = np.min(arrayofTIME[-1, :])
# redostart = find_closest_index_in_range(arrayoftime, univmin, univmin, univmax)
# redostop = find_closest_index_in_range(arrayoftime, univmax, univmin, univmax) + 1
# dim1 = len(newwave)
# dim2 = len(newpos)
# global becon
# becon = {}
# for i in range(0, len(arrayoftime[redostart:redostop])):
#    becon[int(i)] = np.zeros((dim2, dim1))
# for i in range(0, dim2):
#      REBUILD2 = np.zeros((len(Time), len(newwave)))
#      for ii in range(0, len(Time)):
#           REBUILD2[ii] = recon[int(Time[ii])][i, :]
#      for j in range(0, len(newwave)):
#           x = arrayofTIME[:, j]
#           y = REBUILD2[:, j]
#           interp_function = interp1d(x, y, kind='cubic')
#           x_interp = arrayoftime[redostart:redostop]
#           y_interp = interp_function(x_interp)
#           for q in range(0, len(x_interp)):
#               becon[q][i, j] = y_interp[q]
# global newarrayoftime
# newarrayoftime = arrayoftime[redostart:redostop]
# arrayoftime = newarrayoftime
# RTIME = arrayoftime
# Itime = np.array(range(0, len(arrayoftime), 1))
# Time = Itime
# recon = becon

'''
END of chirp correction
'''




# fig, (ax1, ax2, ax3,ax4) = plt.subplots(2,2, figsize=(15, 5))
fig, ((ax1, ax2), (ax3, ax4),(ax5, ax6)) = plt.subplots(3, 2, figsize=(10, 8))

# plt.tight_layout()
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4)

#global slope
#slope = line_equation(pixel1, wave1, pixel2, wave2)[0]
#global interce
#interce = line_equation(pixel1, wave1, pixel2, wave2)[1]
#pix = crop1
#pixvec = np.arange(recon[2].shape[1])
#finalpix = pixvec + pix
# newwave = convert(finalpix,slope,interce)
#newp = np.arange(recon[2].shape[0])
#global subt
#subt = -1 * int(recon[2].shape[0]) / 2
# newpos = convert2(newp,pixelperum,subt)
#global newpos
#newpos = convert2(newp, pixelperum, subt)
#global newwave
#newwave = convert(finalpix, slope, interce)



global corx
corx = N1 - 10

global cory
cory = (crop4 - crop3) - 10

global selected_spots
selected_spots = []  # List to store selected spots: [(corx, cory, wavelength, position), ...]

global shift_pressed
shift_pressed = False  # Track if Shift key is pressed

global show_selected
show_selected = False  # Toggle to show/hide selected spot markers

global select_mode_enabled
select_mode_enabled = False  # Toggle to enable multi-select via left click

valft = RTIME

axT = plt.axes([0.06, 0.96, 0.05, 0.04])
time_textbox = TextBox(ax=axT, label='Time (ps):', initial=f'{RTIME[0]:.2f}\nps (frame 0)')


axT = plt.axes([0.15, 0.97, 0.3, 0.04])
Time_slider = Slider(
    ax=axT,
    label='',
    valmin=Time[0],
    valmax=Time[-1],
    valinit=0,
    valstep=Time,
    valfmt=''

)





global widths
widths = np.zeros(len(Time))
global widthserr
widthserr = np.zeros(len(Time))

global RMSS
RMSS = np.zeros(len(Time))
global RMSSE
RMSSE = np.zeros(len(Time))

global DYATMAX
DYATMAX = np.zeros(len(Time))
global WAVEOFMAX
WAVEOFMAX = np.zeros(len(Time))
woverdub = np.zeros(len(Time))
jay  = np.zeros(len(Time))

# Gaussian-fit parameter storage
global G_AMP, G_MEAN, G_STD, G_OFFSET
G_AMP = np.zeros(len(Time))
G_MEAN = np.zeros(len(Time))
G_STD = np.zeros(len(Time))
G_OFFSET = np.zeros(len(Time))

# Gaussian-fit parameter uncertainties (1-sigma from covariance)
global G_AMP_ERR, G_MEAN_ERR, G_STD_ERR, G_OFFSET_ERR
G_AMP_ERR = np.zeros(len(Time))
G_MEAN_ERR = np.zeros(len(Time))
G_STD_ERR = np.zeros(len(Time))
G_OFFSET_ERR = np.zeros(len(Time))

# Precompute 3D stacks (T, Y, X) and a smoothed version for fast AMPS
global stack, smooth_stack
stack = np.stack([recon[int(t)] for t in Time]).astype(np.float32)
smooth_stack = uniform_filter(stack, size=(1, smthk, smthk), mode='nearest')
# Create TextBox widget
textbox_ax = plt.axes([0.03, 0.8, 0.03, 0.02])  # [left, bottom, width, height]
textbox = TextBox(textbox_ax, 'Max:',initial = 5)
textbox_ax2 = plt.axes([0.03, 0.7, 0.03, 0.02])  # [left, bottom, width, height]
textbox2 = TextBox(textbox_ax2, 'Min:',initial = -3)
exportTA_button_ax = plt.axes([0.1, 0.01, 0.2, 0.05])
exportTA_button = Button(exportTA_button_ax, 'Export This TA', color='grey', hovercolor='0.975')
exportTAM_button_ax = plt.axes([0.3, 0.01, 0.2, 0.05])
exportTAM_button = Button(exportTAM_button_ax, 'Export This TAM', color='grey', hovercolor='0.975')
exportDY_button_ax = plt.axes([0.5, 0.01, 0.2, 0.05])
exportDY_button = Button(exportDY_button_ax, 'Export These Dynamics', color='grey', hovercolor='0.975')
exportFIT_button_ax = plt.axes([0.7, 0.01, 0.2, 0.05])
exportFIT_button = Button(exportFIT_button_ax, 'Export Gaussian Fit', color='grey', hovercolor='0.975')
exportAllDY_button_ax = plt.axes([0.7, 0.07, 0.2, 0.03])
exportAllDY_button = Button(exportAllDY_button_ax, 'Export All These Dynamics', color='grey', hovercolor='0.975')

clearSpots_button_ax = plt.axes([0.5, 0.07, 0.2, 0.03])
clearSpots_button = Button(clearSpots_button_ax, 'Clear Selection', color='lightcoral', hovercolor='red')
showSpots_button_ax = plt.axes([0.3, 0.07, 0.2, 0.03])
showSpots_button = Button(showSpots_button_ax, 'Show Spots', color='lightgrey', hovercolor='0.9')
PAGE1_button_ax = plt.axes([0.15, 0.32, 0.1, 0.02])
PAGE1_button = Button(PAGE1_button_ax, 'Fit', color='grey', hovercolor='Red')
goodpoint_button_ax = plt.axes([0.02, 0.32, 0.1, 0.02])
goodpoint = Button(goodpoint_button_ax, 'Fit here', color='grey', hovercolor='blue')
manual_x_ax = plt.axes([0.12, 0.62, 0.08, 0.03])
manual_y_ax = plt.axes([0.22, 0.62, 0.08, 0.03])
manual_x_tb = TextBox(manual_x_ax, 'X:', initial='')
manual_y_tb = TextBox(manual_y_ax, 'Y:', initial='')
add_spot_ax = plt.axes([0.32, 0.62, 0.1, 0.03])
add_spot_btn = Button(add_spot_ax, 'Add Spot', color='lightgrey', hovercolor='0.9')
fitall_button_ax = plt.axes([0.28, 0.32, 0.1, 0.02])
fitall = Button(fitall_button_ax, 'Try Fit all', color='grey', hovercolor='blue')
fitallmax_button_ax = plt.axes([0.4, 0.32, 0.1, 0.02])
fitallmax = Button(fitallmax_button_ax, 'Try Fit Max', color='grey', hovercolor='blue')
fit_start_idx_ax = plt.axes([0.52, 0.32, 0.1, 0.02])
fit_start_idx = TextBox(fit_start_idx_ax, 'Start i:', initial='0')
fit_ax = plt.axes([0.052, 0.25, 0.025, 0.03])
fit_ax2 = plt.axes([0.052, 0.22, 0.025, 0.03])
fit_ax3 = plt.axes([0.052, 0.19, 0.025, 0.03])
fit_ax4 = plt.axes([0.052, 0.16, 0.025, 0.03])
fit_ax5 = plt.axes([0.052, 0.13, 0.025, 0.03])
fit_ax6 = plt.axes([0.052, 0.1, 0.025, 0.03])
textbox1 = TextBox(fit_ax, 'sinc amp: ',initial=3.3)
textbox3 = TextBox(fit_ax2, 'x off:',initial=-1)
textbox4 = TextBox(fit_ax3, 'sinc wid:',initial=3.2)
textbox5 = TextBox(fit_ax4, 'y off :',initial=0.3)
textbox6 = TextBox(fit_ax5, 'Gauss amp:',initial=-30)
textbox7 = TextBox(fit_ax6, 'Guass wid:',initial=0.68)

num_steps = len(Time)
colormap_name = 'magma'
color_gradient = plt.cm.get_cmap(colormap_name, num_steps)
colors_array = [color_gradient(i / (num_steps - 1)) for i in range(num_steps)]




def submit_text(val):
    ax5.clear()

    for i in range(0, len(Time), 3):
        ax5.scatter(newpos, recon[int(Time[i])][:, int(np.round(corx))], color=colors_array[i], alpha=0.1)
    ax5.scatter(newpos, recon[int(Time_slider.val)][:, int(np.round(corx))], color=colors_array[Time_slider.val])
    # Use Gaussian-only parameters from text boxes:
    # amplitude -> textbox6, mean -> textbox3, stddev -> textbox7, offset -> textbox5
    try:
        amplitude = float(textbox6.text)
    except Exception:
        amplitude = 1.0
    try:
        mean = float(textbox3.text)
    except Exception:
        mean = 0.0
    try:
        stddev = float(textbox7.text)
    except Exception:
        stddev = 1.0
    try:
        offset = float(textbox5.text)
    except Exception:
        offset = 0.0
    ax5.plot(newpos, gaussian_with_offset(newpos, amplitude, mean, stddev, offset), color="red")


def update(val):
    # Update time textbox
    time_textbox.set_val(f'{RTIME[Time_slider.val]:.2f}ps \n(frame {Time_slider.val})')

    # Update plot limits
    plotmax = float(textbox.text)
    plotmin = float(textbox2.text)

    # Calculate AMPS array (vectorized using precomputed smoothed stack)
    try:
        iy = int(np.round(cory))
        ix = int(np.round(corx))
        AMPS = smooth_stack[:, iy, ix]
    except NameError:
        # Fallback: compute on the fly if smooth_stack not available
        AMPS = np.mean([recon[Time[i]][int(np.round(cory)) - 3:int(np.round(cory)) + 3,
                        int(np.round(corx)) - 3:int(np.round(corx)) + 3] for i in range(len(Time))], axis=(1, 2))

    # Update ax4 plot
    ax4.clear()
    ax4.plot(RTIME, AMPS)

    # Update ax1 pcolor plot
    ax1.clear()
    ax1.set_xlabel('pixel(wavelength)')
    ax1.set_ylabel('pixel(space)')
    ax1.set_title("Camera Image")
    ax1.imshow(recon[Time_slider.val], cmap='viridis', vmin=plotmin, vmax=plotmax)
    ax1.axhline(y=cory, color='k', linestyle='--', linewidth=2)
    ax1.axvline(x=corx, color='k', linestyle='--', linewidth=2)
    # Overlay selected spots if toggled on
    if 'show_selected' in globals() and show_selected and 'selected_spots' in globals() and len(selected_spots) > 0:
        xs = [int(s[0]) for s in selected_spots]
        ys = [int(s[1]) for s in selected_spots]
        ax1.scatter(xs, ys, facecolors='none', edgecolors='red', s=80, linewidths=1.5)

    # Update ax2 scatter plot and ax3 line plots
    ax2.clear()
    ax2.set_xlabel('wavelength')
    ax2.set_ylabel('Intensity')
    ax3.clear()
    ax3.set_ylabel('intensity')
    ax3.set_xlabel('space (microns)')
    ax2.set_xlim(newwave.min(), newwave.max())
    ax3.set_xlim(newpos.min(), newpos.max())

    for i in range(0, len(Time), 3):
        ax2.scatter(newwave, recon[int(Time[i])][int(np.round(cory)), :], color=colors_array[i], alpha=0.5)
        ax3.plot(newpos, recon[int(Time[i])][:, int(np.round(corx))], color=colors_array[i], alpha=0.5)

    # Update ax3 average space plot
    avginspace = np.mean(recon[Time_slider.val][:, int(np.round(corx)) - 3:int(np.round(corx)) + 3], axis=1)
    ax3.plot(newpos, avginspace, color="k")

    # Update ax2 and ax3 scatter plots
    ax2.scatter(newwave, recon[Time_slider.val][int(np.round(cory)), :], color="k")
    ax2.axvline(x=newwave[int(np.round(corx))], color='k', linestyle='--', linewidth=2)
    ax3.axvline(x=newpos[int(np.round(cory))], color='k', linestyle='--', linewidth=2)

    # Redraw canvas
    fig.canvas.draw_idle()


#update(0)


def choseloc(event):
    global corx, cory, shift_pressed, selected_spots
    plotmax = textbox.text
    plotmin = textbox2.text
    if event.inaxes == ax1 and event.button == 1:
        corx = (event.xdata)
        cory = (event.ydata)
        # If Shift is held, add to selected spots
        # Check both the tracked state and the event key attribute
        if shift_pressed or (hasattr(event, 'key') and event.key and ('shift' in str(event.key).lower())):
            spot_corx = int(np.round(corx))
            spot_cory = int(np.round(cory))
            wavelength = np.round(newwave[spot_corx])
            position = np.round(newpos[spot_cory])
            spot_info = (spot_corx, spot_cory, wavelength, position)
            # Check if spot already selected
            if spot_info not in selected_spots:
                selected_spots.append(spot_info)
                print(f"Added spot: wavelength={wavelength:.1f}nm, position={position:.1f}um (Total: {len(selected_spots)} spots)")
            else:
                selected_spots.remove(spot_info)
                print(f"Removed spot: wavelength={wavelength:.1f}nm, position={position:.1f}um (Total: {len(selected_spots)} spots)")
    
    # Middle mouse button (button 2) to add/remove spot from selection
    if event.inaxes == ax1 and event.button == 2:
        spot_corx = int(np.round(event.xdata))
        spot_cory = int(np.round(event.ydata))
        wavelength = np.round(newwave[spot_corx])
        position = np.round(newpos[spot_cory])
        spot_info = (spot_corx, spot_cory, wavelength, position)
        # Check if spot already selected
        if spot_info not in selected_spots:
            selected_spots.append(spot_info)
            print(f"Added spot: wavelength={wavelength:.1f}nm, position={position:.1f}um (Total: {len(selected_spots)} spots)")
        else:
            selected_spots.remove(spot_info)
            print(f"Removed spot: wavelength={wavelength:.1f}nm, position={position:.1f}um (Total: {len(selected_spots)} spots)")

    if event.button == 3 and event.inaxes == ax1:  # Check if it's a right-click event
        update(event)
        fig2, ax =  plt.subplots()
        ax.imshow(recon[Time_slider.val], extent = (newwave[0],newwave[-1],newpos[0],newpos[-1]),aspect = 3, cmap='viridis',origin='lower', vmin=plotmin, vmax=plotmax)
        timestamp = 't = {:.1f} ps'.format(round(RTIME[Time_slider.val] - TZERO, 1))
        timeforfile = round(RTIME[Time_slider.val] - TZERO, 1)
        ax.text(np.min(newwave)+5,np.max(newpos) - 2,timestamp,fontsize=20, color='black',bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Space (microns)')
        svg_filename = '{:.1f}ps Cam Image.svg'.format(round(RTIME[Time_slider.val] - TZERO, 1))
        fig2.savefig(svg_filename, format='svg', bbox_inches='tight')
        print("Saved Camera Image")
        plt.close(fig2)
        for i in Time:
            #print((RTIME[i]))
            excel_filename = '{:.1f} ps RAW Cam Image.csv'.format(RTIME[i])
            np.savetxt(excel_filename, recon[i], delimiter=',', fmt='%s')
    if event.button == 3 and event.inaxes == ax2:  # Check if it's a right-click event
        update(event)
        roundlist = np.round(np.geomspace(ITZERO, IETIME, numberofplots)).astype(int)
        num_steps = len(roundlist)
        colormap_name = 'magma'
        color_gradient = plt.cm.get_cmap(colormap_name, num_steps)
        colors_array = [color_gradient(i / (num_steps - 1)) for i in range(num_steps)]
        fig2, ax = plt.subplots()
        for i in range(num_steps):
            ax.scatter(newwave, recon[int(Time[roundlist[i]])][int(np.round(cory)), :],
                       color=colors_array[i], label = '{:.1f} ps'.format(round(RTIME[roundlist[i]] - TZERO, 1)))
        ax.legend()
        ax.set_xlim(newwave.min(), newwave.max())
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel(u'ΔR/R')
        position = np.round(newpos[int(np.round(cory))])
        filename2 = f"TAatpos{position}.svg"
        svg_filename = filename2
        fig2.savefig(svg_filename, format='svg', bbox_inches='tight')
        print("Saved TA Image")
        plt.close(fig2)
    if event.button == 3 and event.inaxes == ax3:  # Check if it's a right-click event
        update(event)
        roundlist = np.round(np.geomspace(ITZERO, IETIME, numberofplots)).astype(int)
        num_steps = len(roundlist)
        colormap_name = 'magma'
        color_gradient = plt.cm.get_cmap(colormap_name, num_steps)
        colors_array = [color_gradient(i / (num_steps - 1)) for i in range(num_steps)]
        fig2, ax = plt.subplots()
        #list(range(9, -1, -1))
        for i in range(num_steps):
            smotth = savgol_filter(recon[int(Time[roundlist[i]])][:, int(np.round(corx))], 3, 1)
            ax.scatter(newpos, smotth, color=colors_array[i], label = '{:.1f} ps'.format(round(RTIME[roundlist[i]] - TZERO, 1)))
        ax.legend()
        ax.set_xlim(newpos.min(), newpos.max())
        ax.invert_yaxis()
        ax.set_xlabel('space (um)')
        ax.set_ylabel(u'ΔR/R')
        wavelength = np.round(newwave[int(np.round(corx))])
        filename2 = f"TAMatwav{wavelength}.svg"
        svg_filename = filename2
        fig2.savefig(svg_filename, format='svg', bbox_inches='tight')
        print("Saved TAM Image")
        plt.close(fig2)
    else:
        # Include the code for other events (if needed)
        pass
    update(event)

def add_manual_spot(event):
    # Read numeric x/y from text boxes and append to selected_spots
    try:
        mx = float(manual_x_tb.text)
        my = float(manual_y_tb.text)
    except Exception:
        print("Invalid X/Y. Please enter numbers.")
        return
    ix = int(np.round(mx))
    iy = int(np.round(my))
    if ix < 0 or iy < 0 or ix >= recon[Time_slider.val].shape[1] or iy >= recon[Time_slider.val].shape[0]:
        print("X/Y out of image bounds.")
        return
    wavelength = np.round(newwave[ix])
    position = np.round(newpos[iy])
    spot_info = (ix, iy, wavelength, position)
    if spot_info not in selected_spots:
        selected_spots.append(spot_info)
        print(f"Added manual spot: wavelength={wavelength:.1f}nm, position={position:.1f}um (Total: {len(selected_spots)} spots)")
    else:
        print("Spot already selected.")
    # trigger redraw to show markers
    update(event)
def update_max(value):
    global plotmax
    plotmax == value
    update(value)

def update_min(value2):
    global plotmin
    plotmin == value2
    update(value2)
def export_to_excel(event):
    exportTA = pd.DataFrame()
    exportTA[f"Wavelength"] = newwave
    for i in Time:
        exportTA[f"{arrayoftime[i]}"] = recon[int(Time[i])][int(np.round(cory)), :]
    position = np.round(newpos[int(np.round(cory))])
    base_filename = os.path.splitext(os.path.basename(filename))[0]
    out_dir = os.path.dirname(filename)
    filename2 = f"{base_filename}TAatpos{position}.xlsx"
    full_path = os.path.join(out_dir, filename2)
    exportTA.to_excel(full_path, index=False)
    print("Done with TA export")
def export_to_excel2(event):
    exportTAM = pd.DataFrame()
    exportTAM[f"Space"] = newpos
    for i in Time:
        exportTAM[f"{arrayoftime[i]}"] = recon[int(Time[i])][:, int(np.round(corx))]
    wavelength = np.round(newwave[int(np.round(corx))])
    base_filename = os.path.splitext(os.path.basename(filename))[0]
    out_dir = os.path.dirname(filename)
    filename2 = f"{base_filename}TAMatwav{wavelength}.xlsx"
    full_path = os.path.join(out_dir, filename2)
    exportTAM.to_excel(full_path, index=False)
    print("Done with TAM export")
def export_to_excel3(event):
    global selected_spots
    
    # If no spots selected, use current spot
    if len(selected_spots) == 0:
        spots_to_export = [(int(np.round(corx)), int(np.round(cory)), 
                           np.round(newwave[int(np.round(corx))]), 
                           np.round(newpos[int(np.round(cory))]))]
        print("No spots selected, exporting current spot only")
    else:
        spots_to_export = selected_spots
        print(f"Exporting {len(spots_to_export)} selected spots")
    
    exportDY = pd.DataFrame()
    exportDY[f"Time"] = arrayoftime
    
    # Export dynamics for each selected spot
    for idx, (spot_corx, spot_cory, wavelength, position) in enumerate(spots_to_export):
        AMPS = np.zeros(len(Time))
        for i in range(0, len(Time), 1):
            # Use the exact clicked pixel, no averaging
            AMPS[i] = recon[Time[i]][spot_cory, spot_corx]
        # Create column name with wavelength and position
        col_name = f"{int(wavelength)}"
        exportDY[col_name] = AMPS
    
    # Export to CSV
    # Extract base filename without extension
    base_filename = os.path.splitext(os.path.basename(filename))[0]
    if len(spots_to_export) == 1:
        wavelength = spots_to_export[0][2]
        position = spots_to_export[0][3]
        filename2 = f"{base_filename}DYatwav{wavelength}pos{position}.csv"
    else:
        filename2 = f"{base_filename}DY_multiple_spots.csv"
    
    out_dir = os.path.dirname(filename)
    full_path = os.path.join(out_dir, filename2)
    exportDY.to_csv(full_path, index=False)
    print(f"Done with Dynamics export: {len(spots_to_export)} spot(s) exported to {filename2}")

def export_to_excel4(event):
    exportW = pd.DataFrame()
    exportW[f"Time"] = arrayoftime
    
    # Export Standard Gaussian fit width (from 'Try Fit all')
    exportW["Standard_Width_um"] = G_STD
    exportW["Standard_Width_Err_um"] = G_STD_ERR
    
    # Export Max Fit Gaussian width (from 'Try Fit Max')
    # Check if widths array is populated
    if 'widths' in globals() and np.any(widths):
        exportW["MaxFit_Width"] = widths
        exportW["MaxFit_Width_Err"] = widthserr
        
    wavelength = np.round(newwave[int(np.round(corx))])
    base_filename = os.path.splitext(os.path.basename(filename))[0]
    out_dir = os.path.dirname(filename)
    filename5 = f"{base_filename}GaussianW_atwav{wavelength}.xlsx"
    full_path = os.path.join(out_dir, filename5)
    exportW.to_excel(full_path, index=False)
    print("Done with Gaussian fit export")

def export_to_excel_all_dynamics(event):
    global selected_spots
    
    # Determine spots to export
    if len(selected_spots) == 0:
        spots_to_export = [(int(np.round(corx)), int(np.round(cory)), 
                           np.round(newwave[int(np.round(corx))]), 
                           np.round(newpos[int(np.round(cory))]))]
        print("No spots selected, exporting current spot only for ALL files")
    else:
        spots_to_export = selected_spots
        print(f"Exporting {len(spots_to_export)} selected spots for ALL files")

    # Find all *sum.txt files in the directory
    search_pattern = os.path.join(os.path.dirname(filename), "*sum.txt")
    sum_files = glob.glob(search_pattern)
    
    if not sum_files:
        print("No *sum.txt files found.")
        return

    print(f"Found {len(sum_files)} files to process...")

    # Extract time and intensity from each file separately and combine side-by-side
    # No alignment - each file keeps its own time axis
    all_columns = {}  # column_name -> list of values
    max_rows = 0
    
    for f_idx, f_path in enumerate(sum_files):
        try:
            base_name = os.path.basename(f_path)
            para_path = f_path.replace("sum.txt", "para.txt")
            
            if not os.path.exists(para_path):
                print(f"Warning: Parameter file not found for {base_name}, skipping.")
                continue
            
            # Load Image
            try:
                curr_images = pd.read_csv(f_path, delimiter='\t', header=None)
            except Exception:
                curr_images = pd.read_csv(f_path, header=None)
            
            curr_images[curr_images == '--'] = np.nan
            curr_images = curr_images.values
            
            # Load Time
            curr_time_df = pd.read_csv(para_path, delimiter='\t')
            curr_time_df.columns = curr_time_df.columns.astype(float)
            curr_time_arr = np.array(curr_time_df.columns)
            
            # Reconstruct images with smoothing
            curr_recon = {}
            curr_Itime = np.arange(len(curr_time_arr))
            
            for i in curr_Itime:
                leng = i * N2
                if leng + N2 > curr_images.shape[0]:
                    break
                raw = np.rot90(curr_images[leng + crop1:leng + N2 - crop2, N1 - crop4:N1 - crop3])
                smoothed_array = uniform_filter(raw.astype(np.float32), size=smthk, mode='nearest')
                curr_recon[i] = smoothed_array

            # Extract dynamics for each selected spot
            for s_idx, (spot_corx, spot_cory, spot_wav, spot_pos) in enumerate(spots_to_export):
                # Extract time and intensity arrays for this file/spot
                times = []
                intensities = []
                
                for i in curr_Itime:
                    if i in curr_recon:
                        times.append(curr_time_arr[i])
                        intensities.append(curr_recon[i][spot_cory, spot_corx])
                
                # Create column names for this file/spot
                file_prefix = base_name.replace('sum.txt','')
                time_col = f"{file_prefix}_Time_W{int(spot_wav)}_P{int(spot_pos)}"
                intensity_col = f"{file_prefix}_Intensity_W{int(spot_wav)}_P{int(spot_pos)}"
                
                # Store in dictionary
                all_columns[time_col] = times
                all_columns[intensity_col] = intensities
                
                # Track max rows for padding
                max_rows = max(max_rows, len(times))
            
            print(f"  Processed {base_name}")

        except Exception as e:
            print(f"Error processing {f_path}: {e}")
            continue

    if not all_columns:
        print("No data extracted from any files.")
        return
    
    # Pad all columns to same length with NaN
    for col_name in all_columns:
        col_len = len(all_columns[col_name])
        if col_len < max_rows:
            all_columns[col_name].extend([np.nan] * (max_rows - col_len))
    
    # Create DataFrame from dictionary
    combined_df = pd.DataFrame(all_columns)
    
    # Export
    out_dir = os.path.dirname(filename)
    
    # Add wavelength to filename to avoid overwriting
    if len(spots_to_export) > 0:
        first_wav = int(spots_to_export[0][2])
        out_name = f"AllDynamics_Combined_W{first_wav}.csv"
    else:
        out_name = "AllDynamics_Combined.csv"
        
    full_path = os.path.join(out_dir, out_name)
    combined_df.to_csv(full_path, index=False)
    print(f"Batch export complete! Saved to {out_name} with {len(combined_df)} rows")


def switchpage(event):
        # Gaussian-only fit on current time's spatial profile
        why = np.mean(recon[Time_slider.val][:, int(np.round(corx)) - 1:int(np.round(corx)) + 1], axis=1)
        # Initial guesses for Gaussian with offset
        amp0 = why[np.argmax(np.abs(why))]
        mu0 = newpos[np.argmax(np.abs(why))]
        sigma0 = max((newpos[-1] - newpos[0]) / 10.0, 1e-3)
        offset0 = np.median(why)
        p0 = [amp0, mu0, sigma0, offset0]
        params, covariance = curve_fit(gaussian_with_offset, newpos, why, p0=p0, maxfev=20000)
        A_fit, mu_fit, sigma_fit, offset_fit = params
        # Map Gaussian params to existing textboxes:
        # textbox6 -> amplitude, textbox3 -> mean, textbox7 -> stddev, textbox5 -> offset
        textbox6.set_val(np.round(A_fit,2))
        textbox3.set_val(np.round(mu_fit,2))
        textbox7.set_val(np.round(np.abs(sigma_fit),2))
        textbox5.set_val(np.round(offset_fit,2))

    #submit_text(event)
        #choseloc(event)

def addpoint(event):
    ax5.clear()
    ax6.clear()
    submit_text(event)
    #A_fit, o_fit, w_fit, c_fit, B_fit, J_fit = params
    #errors = np.sqrt(np.diag(covariance))
    #widths[Time_slider.val] = J_fit*J_fit
    #widthserr[Time_slider.val] = errors[5]*errors[5]
    #ax6.clear()
    #ax1.set_xlabel('Time')
    #ax1.set_ylabel('RMS')
    ##ax6.errorbar(arrayoftime,widths ,yerr=widthserr, fmt='bo', label='Data with Error Bars')
    ##ax6.plot(arrayoftime,widths)
    #rmser = calculate_rms_error(newpos,params,errors)
    #RMSSE[Time_slider.val] = 2 * rmser
    ##rmser)
    ## Specify the range for integration and plotting
    #L = newpos[-1]-newpos[0]
    #xforthis = np.linspace(newpos[0],newpos[-1],5000)
    #yvalforrms = sincplusgauss(xforthis, A_fit, 0, w_fit, 0, B_fit, J_fit)
    #result_line1, _ = quad(integrand_sincplusgauss_squared, -L, L, args=(A_fit, 0, w_fit, 0, B_fit, J_fit))
    #rms_value_line1 = -1 * np.sqrt(result_line1 / (2 * L))
    ##print(w_fit**2)
    #tolerance = 0.01  # Adjust the tolerance based on your needs
    #indices = xforthis[np.where(np.abs(yvalforrms - rms_value_line1) < tolerance)[0]]
    #indices = indices[indices >= 0]
    ##print(rms_value_line1)

    #RMSS[Time_slider.val] = indices[0]**2
    #nonzeroinx = np.where(np.array(RMSS) > 0)[0]
    #filtered_arrayoftime = np.array(arrayoftime)[nonzeroinx]
    #filtered_RMSSE = np.array(RMSSE)[nonzeroinx]
    #filtered_RMSS = np.array(RMSS)[nonzeroinx]
    #ax6.errorbar(filtered_arrayoftime, filtered_RMSS, yerr=filtered_RMSSE, fmt='o', label='Data with Error Bars')

    # Scatter plot without error bars
    #ax6.scatter(filtered_arrayoftime, filtered_RMSS, color='red', marker='o', label='Scatter Plot')

def fitallfunc(event):
    # Fit Gaussian only to spatial profiles at each time and print parameters
    # Determine starting frame from textbox (accept index or time in ps)
    try:
        raw_txt = str(fit_start_idx.text).strip().lower().replace('ps', '')
        start_i = int(float(raw_txt)) # Force to be frame index
    except Exception:
        start_i = 0
    start_i = max(0, min(len(Time) - 1, int(start_i)))
    for i in range(start_i, len(Time), 1):
        try:
            # Spatial line-out around current corx
            why = np.mean(recon[i][:, int(np.round(corx)) - 3:int(np.round(corx)) + 3], axis=1)
            # Initial guesses for Gaussian with offset
            amp0 = why[np.argmax(np.abs(why))]
            mu0 = newpos[np.argmax(np.abs(why))]
            sigma0 = max((newpos[-1] - newpos[0]) / 10.0, 1e-3)
            offset0 = np.median(why)
            p0 = [amp0, mu0, sigma0, offset0]
            params, covariance = curve_fit(gaussian_with_offset, newpos, why, p0=p0, maxfev=20000)
            A_fit, mu_fit, sigma_fit, offset_fit = params
            # Store results
            G_AMP[i] = A_fit
            G_MEAN[i] = mu_fit
            G_STD[i] = np.abs(sigma_fit)
            G_OFFSET[i] = offset_fit
            # Store 1-sigma uncertainties if available
            try:
                errs = np.sqrt(np.diag(covariance))
                G_AMP_ERR[i] = errs[0]
                G_MEAN_ERR[i] = errs[1]
                G_STD_ERR[i] = np.abs(errs[2])
                G_OFFSET_ERR[i] = errs[3]
            except Exception:
                G_AMP_ERR[i] = np.nan
                G_MEAN_ERR[i] = np.nan
                G_STD_ERR[i] = np.nan
                G_OFFSET_ERR[i] = np.nan
            # Print Gaussian fit parameters
            if np.isfinite(G_STD_ERR[i]) and G_STD_ERR[i] > 0:
                print(f"t={arrayoftime[i]:.3f} ps | A={A_fit:.4g}±{G_AMP_ERR[i]:.2g}, mean={mu_fit:.4g}±{G_MEAN_ERR[i]:.2g} um, std={np.abs(sigma_fit):.4g}±{G_STD_ERR[i]:.2g} um, offset={offset_fit:.4g}±{G_OFFSET_ERR[i]:.2g}")
            else:
                print(f"t={arrayoftime[i]:.3f} ps | A={A_fit:.4g}, mean={mu_fit:.4g} um, std={np.abs(sigma_fit):.4g} um, offset={offset_fit:.4g}")
        except RuntimeError:
            print(f"t={arrayoftime[i]:.3f} ps | Gaussian fit did not converge")
            continue
        except Exception as e:
            print(f"t={arrayoftime[i]:.3f} ps | Fit error: {e}")
            continue

    # Plot Gaussian stddev vs time
    ax6.clear()
    ax6.set_xlabel('Time (ps)')
    ax6.set_ylabel('Gaussian std (um)')
    # Plot results starting from the chosen start frame, with error bars
    ax6.errorbar(arrayoftime[start_i:], G_STD[start_i:], yerr=G_STD_ERR[start_i:], fmt='o-', capsize=3)
    ax6.set_title('Gaussian width vs time')
    ax6.figure.canvas.draw_idle()

def fitallfuncmax(event):
    # Determine starting frame from textbox (accept index or time in ps)
    try:
        raw_txt = str(fit_start_idx.text).strip().lower().replace('ps', '')
        start_i = int(float(raw_txt)) # Force to be frame index
    except Exception:
        start_i = 0
    start_i = max(0, min(len(Time) - 1, int(start_i)))
    for i in range(start_i, len(Time), 1):
        try:
            A = float(textbox1.text)
            o = float(textbox3.text)
            w = float(textbox4.text)
            c = float(textbox5.text)
            B = float(textbox6.text)
            J = float(textbox7.text)
            #print(corx)
            #potmax = np.abs(recon[i][:, int(np.round(corx)) - 10:int(np.round(corx)) + 10])
            #print(potmax)
            windowsizeforcheck = 10
            maxvalue = np.max(np.abs(recon[i][:, int(np.round(corx)) -windowsizeforcheck:int(np.round(corx)) + windowsizeforcheck]))
            DYATMAX[i] = maxvalue
            indices = np.argmax(np.abs(recon[i][int(np.round(cory)), int(np.round(corx)) -windowsizeforcheck:int(np.round(corx)) + windowsizeforcheck]))
            maximidx = int(np.round(corx)) + indices - windowsizeforcheck
            maxr = newwave[maximidx]
            WAVEOFMAX[i] = maxr

            why = np.mean(recon[i][:, int(np.round(maximidx)) - 5:int(np.round(maximidx)) + 5], axis=1)
            penut = [A, o, w, c, B, J]
            params, covariance = curve_fit(sincplusgauss, newpos, why, p0=penut)
            A_fit, o_fit, w_fit, c_fit, B_fit, J_fit = params
            errors = np.sqrt(np.diag(covariance))
            widths[i] = J_fit * J_fit
            widthserr[i] = errors[5] * errors[5]

            rmser = calculate_rms_error(newpos, params, errors)
            RMSSE[i] = 2 * rmser
            # rmser)
            # Specify the range for integration and plotting
            L = newpos[-1] - newpos[0]
            xforthis = np.linspace(newpos[0], newpos[-1], 5000)
            yvalforrms = sincplusgauss(xforthis, A_fit, 0, w_fit, 0, B_fit, J_fit)
            result_line1, _ = quad(integrand_sincplusgauss_squared, -L, L, args=(A_fit, 0, w_fit, 0, B_fit, J_fit))
            rms_value_line1 = -1 * np.sqrt(result_line1 / (2 * L))
            #print(J_fit**2)
            tolerance = 0.01  # Adjust the tolerance based on your needs
            indices = xforthis[np.where(np.abs(yvalforrms - rms_value_line1) < tolerance)[0]]
            indices = indices[indices >= 0]
            # print(indices)

            RMSS[i] = indices[0]**2

            nonzeroinx = np.where(np.array(RMSS) > 0)[0]
            filtered_arrayoftime = np.array(arrayoftime)[nonzeroinx]
            filtered_RMSSE = np.array(RMSSE)[nonzeroinx]
            filtered_RMSS = np.array(RMSS)[nonzeroinx]

        except RuntimeError:
            print("fit did not converge")
            pass
        except IndexError:
            pass
    #print(WAVEOFMAX)
    #print(DYATMAX)
    # Plot Gaussian Width vs Time (instead of RMS)
    ax6.clear()
    ax6.set_xlabel('Time (ps)')
    ax6.set_ylabel('Gaussian Width')
    
    # Filter out zeros (where fit wasn't run or failed)
    nonzeroinx = np.where(np.array(widths) > 0)[0]
    filtered_time = np.array(arrayoftime)[nonzeroinx]
    filtered_widths = np.array(widths)[nonzeroinx]
    filtered_errors = np.array(widthserr)[nonzeroinx]
    
    # Plot with error bars
    ax6.errorbar(filtered_time, filtered_widths, yerr=filtered_errors, fmt='o-', capsize=3, label='Width')
    ax6.set_title('Gaussian Width vs Time (Max Fit)')
    ax6.legend()
    ax6.figure.canvas.draw_idle()
def changetime(event):
    global shift_pressed
    if event.key == 'right':
        Time_slider.set_val(min(Time_slider.val + 1, Time_slider.valmax))  # Increment the slider value by 1
    elif event.key == 'left':
        Time_slider.set_val(max(Time_slider.val - 1, Time_slider.valmin))
    elif event.key == 'shift':
        shift_pressed = True
    update(event)

def on_key_release(event):
    global shift_pressed
    if event.key == 'shift':
        shift_pressed = False

def clear_selection(event):
    global selected_spots
    selected_spots = []
    print("Selection cleared")

def toggle_show_spots(event):
    global show_selected
    show_selected = not show_selected
    # Toggle button label
    if show_selected:
        showSpots_button.label.set_text('Hide Spots')
    else:
        showSpots_button.label.set_text('Show Spots')
    fig.canvas.draw_idle()
    update(event)

fig.canvas.mpl_connect('button_press_event', choseloc)
fig.canvas.mpl_connect('key_press_event', changetime)
fig.canvas.mpl_connect('key_release_event', on_key_release)

Time_slider.on_changed(update)
textbox.on_submit(update_max)
textbox2.on_submit(update_min)
exportTA_button.on_clicked(export_to_excel)
exportTAM_button.on_clicked(export_to_excel2)
exportDY_button.on_clicked(export_to_excel3)
exportFIT_button.on_clicked(export_to_excel4)
clearSpots_button.on_clicked(clear_selection)
showSpots_button.on_clicked(toggle_show_spots)
PAGE1_button.on_clicked(switchpage)
textbox1.on_submit(submit_text)
textbox3.on_submit(submit_text)
textbox4.on_submit(submit_text)
textbox5.on_submit(submit_text)
textbox6.on_submit(submit_text)
textbox7.on_submit(submit_text)
goodpoint.on_clicked(addpoint)
fitall.on_clicked(fitallfunc)
fitallmax.on_clicked(fitallfuncmax)
exportAllDY_button.on_clicked(export_to_excel_all_dynamics)
add_spot_btn.on_clicked(add_manual_spot)



plt.draw()
plt.show()




