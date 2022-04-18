# -*- coding: utf-8 -*-
"""
This module is a GUI and provides additional functionality for Ron Sprouse's klsyn Python interface (https://github.com/rsprouse/klsyn)
which is itself a port of Dennis Klatt's original C speech synthesizer system. 
updated: March 24, 2022
author: Eric Wilbanks
"""
import pandas as pd
import pandastable as ps
import os
import matplotlib.pyplot as plt
import audiolabel as al
import numpy as np
import tkinter as tk
import pygame as pg

from scipy.io import wavfile
from sklearn.linear_model import LinearRegression
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import MouseButton
from matplotlib.widgets import RectangleSelector
from klsyn import klpfile
import klsyn.klatt_wrap as klatt_wrap

# this import is just required for custom_read;
# TO-DO: remove when custom_read is legacied
import re

def custom_read(fname,*args):
    ''' Read a .klp parameter file into a dict and return the dict. Also return comments.
    This is a slightly edited version of klsyn.klpfile.read(). 
    I've suggested these updates as a pull request, and will remove this if/when they are approved.
    '''
    sep = "\s+"
    params = {}
    comments = {'header': '', 'constant': {}, 'varied': []}
    fields = None
    field_map = {}
    varparams_re = re.compile('^\s*_varied_params_\s*$')
    comment_re = re.compile('^\s*#')               # a comment line
    empty_re = re.compile('^\s*$')                 # an empty line
    eol_comment_re = re.compile('(?P<comment>\s*#.*)$')         # an end-of-line comment
    with open(fname, 'r') as f:

        # Read header comments.
        reading_header = True
        header_comments = ''
        loc = f.tell()
        while reading_header:
            line = f.readline()
            if comment_re.search(line):
                loc = f.tell()
                header_comments += line.rstrip() + "\n"
            else:
                f.seek(loc)   # Not a comment, rewind to previous line.
                comments['header'] = header_comments
                reading_header = False

        # Read constant and varied parameters.
        reading_constparams = True
        for line in f.readlines():
            if varparams_re.search(line):
                reading_constparams = False
                continue
            elif empty_re.search(line):
                continue
            elif reading_constparams:
                comment = ''
                m = eol_comment_re.search(line)
                if m:
                    comment = m.group('comment')
                    line = eol_comment_re.sub('', line)
                (p, val) = re.split(sep, line.strip())
                params[p.strip()] = int(round(float(val.strip())))
                comments['constant'][p.strip()] = comment.rstrip()
            elif fields == None:       # reading_constparams == False
                line = eol_comment_re.sub('', line)
                fields = re.split(sep, line.strip())
                for idx,fld in enumerate(fields):
                    fld = fld.strip()
                    field_map[str(idx)] = fld
                    if fld in klatt_wrap.params_map.keys():
                        params[fld] = []
                    elif (fld.startswith('_') and fld.endswith('_')):
                        if (args and fld in args):
                            params[fld] = []
                    elif not (fld.startswith('_') and fld.endswith('_')):
                        raise Exception(
                            "Unrecognized varied parameter '{:s}'.\n".format(
                                fld)
                            )
            else:
                comment = ''
                m = eol_comment_re.search(line)
                if m:
                    comment = m.group('comment')
                    line = eol_comment_re.sub('', line)
                comments['varied'].append(comment.rstrip())
                vals = re.split(sep, line.strip())
                for idx,val in enumerate(vals):
                    val = val.strip()
                    fld = field_map[str(idx)]
                    if fld in klatt_wrap.params_map.keys():
                        params[fld].append(int(round(float(val))))
                    elif (args and fld in args):
                        params[fld].append(int(round(float(val))))
    return (params, comments)

# make sure to update references to class attributes if you change the class name..
class klp_gui():
    # these klatt parameters are never allowed to be variable
    # min, default, and max values are as described in Klatt & Klatt (1990) - JASA, Table XI
    always_fixed = {'sr': {'min': 5000, 'default': 10000, 'max': 20000}, # "output sampling rate, in sample/s"
        'nf': {'min': 1, 'default': 5, 'max': 6}, # "number of formants in the cascade branch"
        'du': {'min': 30, 'default': 500, 'max': 5000}, # "duration of the utterance, in ms"
        'ss': {'min': 1, 'default': 2, 'max': 3}, # "source switch (1 = impulse, 2 = natural, 3 = LF model)"
        'ui': {'min': 1, 'default': 5, 'max': 20}, # "update interval for the parameter reset, in ms"
        'rs': {'min': 1, 'default': 8, 'max': 8191}, # "random seed (initial value of the random number generator"
        'os': {'min': 0, 'default': 0, 'max': 20} # "output selector (0 = normal, 1 = voicing source, ..."
        }


    def __init__(self, wav_path, tg_path, klp_params_path):
        """
        Arguments:
            wavpath: path to wav file
            tg_path: path to tg; Minimum requirement of columns for start time ['t1'], end time ['t2'], and phone label ['label']
            klp_params_path: path to klp parameters file
        """
        # process wav and klp files
        self.__process_wav_klp(wav_path, klp_params_path)
        
        # process tg_path
        [self.tg] = al.read_label(tg_path, ftype='praat')
        self.tg['t1'] = self.tg['t1'] * 1000 # convert to ms
        self.tg['t2'] = self.tg['t2'] * 1000 # convert to ms
        self.__add_tg_labels() # add tg info to variable df
        
        # default options
        self.win_size = '960x960'
        self.popup_size = '600x300'
        self.bpad = 5 # button padding default
        self.outlier_threshold = 1.75 # residuals threshold for detecting outliers
        self.params_wav_unsyncced = False # to track if wdata is out of date with klp params
        self.topframe = None
        self.current_toggle = None
        self.dragging_point = None
        self.dragging_point_ix = None
        self.dragging_point_artist = None

    def __process_wav_klp(self,wav_file,klp_file):
        # process wav_path
        self.sr, self.wdata, self.wtimes = self.process_wav(wav_file)
        self.channels = 1 # assume mono input

        # process klp_params_path
        # TO-DO: change to klpfile.read() when custom_read is legacied
        self.klp_params, self.klp_comments = custom_read(klp_file,'_msec_')#klpfile.read(klp_params_path,'_msec_')
        # TO-DO: I think we just need two types, not 3; review all uses and confirm
        self.base_params = pd.DataFrame.from_dict(self.klp_params) # base params which maps to the current wav; this is updated when new wav is generated
        self.working_params = self.base_params.copy() # copy of working params, used to detect changes of toplevel self.params
        self.params = self.base_params.copy() # top level df, safe to edit/output

    def start(self):
        # create window
        self.__initialize_frame()
        
        # run process
        self.root.mainloop()
        
        # close last plot
        plt.close()

    def __initialize_frame(self):
        # build root window
        self.root = tk.Tk()
        self.root.geometry(self.win_size)
        self.default_font = tk.font.Font(root=self.root, family='bitstream charter', size = 14)
        self.root.option_add('*font',self.default_font)

        # build options frame
        self.opts_frame = tk.Frame(self.root)
        
        # header instructions
        tk.ttk.Label(self.opts_frame, text = 'Select variables to plot:', justify = 'center').pack(padx=self.bpad, pady=self.bpad)

        # add parameter container frame
        self.param_container = tk.Frame(self.opts_frame, borderwidth=5, relief=tk.GROOVE)
        self.param_container.pack(side=tk.TOP, fill='both', expand=True)
        
        # set expanding behavior of param_container's inner grid
        self.param_container.grid_columnconfigure(0, weight=1)
        self.param_container.grid_columnconfigure(1, weight=1) # setting equal weight (> 0) to both columns leads to them expanding to half
        self.param_container.grid_rowconfigure(0, weight=1)
        
        # add fixed and variable frames
        self.fixed_frame = tk.Frame(self.param_container, borderwidth=5, relief=tk.GROOVE, background='#F7AEF8')
        self.fixed_frame.grid(row=0,column=0,sticky='NSEW')
        
        self.variable_frame = tk.Frame(self.param_container, borderwidth=5, relief=tk.GROOVE, background='#72DDF7')
        self.variable_frame.grid(row=0,column=1,sticky='NSEW') 
        
        self.__init_parameters()
        self.__neat_buttons(self.fixed_frame,5)
        self.__neat_buttons(self.variable_frame,5)

        # build nav_container and buttons
        nav_container = tk.Frame(self.root, borderwidth=5, background='#B388EB', relief=tk.GROOVE)

        nav0 = tk.Button(nav_container, text = 'Choose Variable(s)', command = self.__move_to_opts)
        nav1 = tk.Button(nav_container, text = 'Plot', command = self.__move_to_plot)
        nav2 = tk.Button(nav_container, text = 'Set Fixed value', command = self.__set_fixed_value)
        nav3 = tk.Button(nav_container, text = 'Run new synthesis', command = self.__new_synthesis)
        nav4 = tk.Button(nav_container, text = 'Play Audio', command = self.play)
        nav5 = tk.Button(nav_container, text = 'Show DF', command = self.__open_df)
        nav6 = tk.Button(nav_container, text = 'Close GUI', command = self.root.destroy)
        
        nav_buttons = {button['text']: button for button in [nav0,nav1,nav2,nav3,nav4,nav5,nav6]}
        self.__neat_buttons(nav_container,3,nav_buttons)

        # set button background colors
        self.nottoggled_color = nav0.cget('background')
        self.nottoggled_color_hover = nav0.cget('activebackground')
        self.toggled_color = '#5CE497'
        self.toggled_color_hover = '#88EBB3'

        # build plot frame
        self.plot_frame = tk.Frame(self.root)    
        self.__move_to_opts()
        
        # add button container
        nav_container.pack(side=tk.BOTTOM, fill=tk.X)

    def __init_parameters(self):
        self.param_types = {}
        self.param_booleans = {}
        self.checkbuttons = {}
        # sort parameters into fixed and variable, then create buttons
        for param in sorted(self.params.columns):
            if param not in ['_msec_', 'label']:
                self.param_booleans[param] = tk.BooleanVar()
        self.__check_parameter_fixedvariable_status()

    def __check_parameter_fixedvariable_status(self):
        for param in sorted(self.param_booleans.keys()):
            if param in klp_gui.always_fixed.keys(): # check that parameters that should never vary are actually fixed
                if self.params[param].nunique() > 1:
                    progress_var = self.__popup('\n\n'.join(['ERROR - the following parameter should never vary.','Please correct it immediately or prepare for failure: ',param]), progress = True)
                    # don't progress until we've made a choice
                    # to-do: test to make sure closing the window early won't cause program to hang infinitely
                    self.root.wait_variable(progress_var)
                    self.__popup('to-do: have this be a selection!')  # to-do, have this be a selection to force a specific value
                self.param_types[param] = 'always_fixed'
                #self.checkbuttons[param] = tk.Checkbutton(self.fixed_frame, variable = self.param_booleans[param], text = param, relief = tk.SOLID, overrelief = 'sunken', cursor = 'spider')
            else:
                if self.params[param].nunique() > 1:
                    self.param_types[param] = 'variable'
                    self.checkbuttons[param] = tk.Checkbutton(self.variable_frame, variable = self.param_booleans[param], text = param, relief = tk.SOLID, overrelief = 'sunken')
                elif self.params[param].nunique() == 1:
                    self.param_types[param] = 'fixed'
                    self.checkbuttons[param] = tk.Checkbutton(self.fixed_frame, variable = self.param_booleans[param], text = param, relief = tk.SOLID, overrelief = 'sunken')
        self.__neat_buttons(self.fixed_frame,5)
        self.__neat_buttons(self.variable_frame,5)
            
    def __neat_buttons(self, containing_frame, ncol, buttons_dict = None):
        # padding for if we have a label or not
        num_labels = 0
        # determine if the buttons we're placing are fixed, variable, or generic type
        if containing_frame == self.variable_frame:
            buttons_dict = {param:button for param, button in self.checkbuttons.items() if self.param_types[param] == 'variable'}
            # set label
            variable_label = tk.ttk.Label(containing_frame, text='Variable Parameters:', justify='center', borderwidth = 3, relief = 'ridge')
            variable_label.grid(row=0,column=0,columnspan=ncol,padx=self.bpad,pady=self.bpad)
            num_labels = 1
        elif containing_frame == self.fixed_frame:
            buttons_dict = {param:button for param, button in self.checkbuttons.items() if self.param_types[param] == 'fixed'}
            # set label
            fixed_label = tk.ttk.Label(containing_frame, text='Fixed Parameters:', justify='center', borderwidth = 3, relief = 'ridge')
            fixed_label.grid(row=0,column=0,columnspan=ncol,padx=self.bpad,pady=self.bpad)
            num_labels = 1

        # calculate layout
        total = len(buttons_dict.keys())
        full_rows, remainder = divmod(total, ncol)
        if remainder > 0:
            nrow = full_rows + 1
        else:
            nrow = full_rows

        # grid weight config
        for c in range(ncol):
            containing_frame.grid_columnconfigure(c, weight=1)
        for r in range(nrow+num_labels):
            if (r > 0) or (num_labels == 0): # ignoring label rows
                containing_frame.grid_rowconfigure(r, weight=1)
        # place buttons in grid
        for ix, (param, button) in enumerate(buttons_dict.items()):
            row_n, col_n  = divmod(ix, ncol)
            # adding +1 to row_n to account for the labels
            button.grid(row=row_n+num_labels,column=col_n,sticky='NSEW',padx=self.bpad,pady=self.bpad)
            
    def __open_df(self):
        # focus dfwin if it exists when button is pressed
        if hasattr(self,'dfwin'):
            self.dfwin.lift()
        
        # otherwise create the window
        else: 
            # initialize window 
            self.dfwin = tk.Toplevel()
            self.dfwin.wm_title('Double-Click to Edit - All Changes Saved on Window exit')
            self.dfwin.geometry('600x400')
            
            # map non-button exits to close_df() function to avoid errors
            self.dfwin.protocol('WM_DELETE_WINDOW', self.__close_df)
            
            # nested window for data
            df_box = tk.Frame(self.dfwin)
            df_box.pack(side=tk.TOP,fill='both', expand=1)
            
            # attach Pandastable Table to self.dfwin
            self.dft = ps.Table(df_box, dataframe = self.params, showstatusbar = True)
            self.dft.show()
            b1 = tk.Button(self.dfwin, text = 'Close', command = self.__close_df).pack(side=tk.BOTTOM)
    
    def __close_df(self):
        # monitor df changes
        self.__update_df()
        
        # destroy dfwin frame and remove from instance attributes
        self.dfwin.destroy()
        delattr(self,'dft')
        delattr(self,'dfwin')

    def __update_frames(self):
        if self.topframe == self.opts_frame:
            self.__move_to_opts()
        elif self.topframe == self.plot_frame:
            self.__move_to_plot()    
            
    def __update_df(self,update_frames = True):
        # flag if the base df has changed
        if not self.params.equals(self.working_params):
            self.params_wav_unsyncced = True # only reset if we read in a new set of syncced wavdata

            # reassign parameters to catch any fixed/variable that have changed type
            self.__check_parameter_fixedvariable_status()
            
            # update working_params
            self.working_params = self.params.copy()
        
            # refresh active frame
            if update_frames == True:
                self.__update_frames()
            
            # update dft in case we're editing somewhere else but the Pandastable widget is active
            if hasattr(self,'dft'):
                self.dft.updateWidgets()

    def __set_fixed_value(self):
        chosen_param = self.__param_choice_window('all_fixed')
        self.__value_choice_window(chosen_param)

    def __param_choice_window(self, param_type = None, specified = None):
        # initialize window 
        set_fixed_win = tk.Toplevel()
        set_fixed_win.wm_title('Choose which fixed variable to overwrite')
        set_fixed_win.geometry(self.popup_size)
                
        choice = tk.StringVar(set_fixed_win)
        bs = []

        # default to choose from all
        if param_type is None:
            for param, value in self.param_types.items():
                bs.append(tk.Radiobutton(set_fixed_win, text = param, variable = choice, value = param, command = set_fixed_win.destroy))
        # only fixed
        elif param_type == 'fixed':
            for param, value in self.param_types.items():
                if value == 'fixed':
                    bs.append(tk.Radiobutton(set_fixed_win, text = param, variable = choice, value = param, command = set_fixed_win.destroy))
        # 'fixed' and 'always_fixed' types
        elif param_type == 'all_fixed':
            for param, value in self.param_types.items():
                if value == 'fixed' or value == 'always_fixed':
                    bs.append(tk.Radiobutton(set_fixed_win, text = param, variable = choice, value = param, command = set_fixed_win.destroy))
        # 'variable'
        elif param_type == 'variable':
            for param, value in self.param_types.items():
                if value == 'fixed' or value == 'always_fixed':
                    bs.append(tk.Radiobutton(set_fixed_win, text = param, variable = choice, value = param, command = set_fixed_win.destroy))
        # currently plotted
        elif param_type == 'plotted':
            for param in self.data_artists.keys():
                bs.append(tk.Radiobutton(set_fixed_win, text = param, variable = choice, value = param, command = set_fixed_win.destroy))
        # list of specified params
        elif param_type == 'specified' and specified is not None:
            for param in specified:
                bs.append(tk.Radiobutton(set_fixed_win, text = param, variable = choice, value = param, command = set_fixed_win.destroy))
        
        # arrange buttons
        buttons = {button['text']: button for button in bs}
        self.__neat_buttons(set_fixed_win, 4, buttons)
                    
        # don't progress until we've made a choice
        # to-do: test to make sure closing the window early won't cause program to hang infinitely
        self.root.wait_variable(choice)

        return choice.get()

    def __value_choice_window(self,chosen_param,limits=None):
        """
        Limits is a tuple of form (t1,t2), marking the time range to replace with the new fixed value
        If no limits are given, all values will be chosen.
        """
        # choice window
        choose_value_win = tk.Toplevel()
        choose_value_win.wm_title('Set new value (INT)')
        choose_value_win.geometry(self.popup_size)

        input_value = tk.StringVar(choose_value_win)
        new_value = tk.IntVar()
        entry = tk.Entry(choose_value_win, textvariable = input_value)
        confirm = tk.Button(choose_value_win, text = 'Confirm', command = lambda: self.__validate_int(choose_value_win,input_value,new_value))

        entry.grid(row=0,column=1)
        confirm.grid(row=3,column=1)
        self.root.wait_variable(new_value)

        # confirmation window
        if limits is None:
            old_value = self.params[chosen_param].unique().tolist()[0] # hard-coding position, assuming that we've correctly only allowing 1 value for fixed/always_fixed parameters
            self.params[chosen_param] = new_value.get()
            self.__update_df()
            self.__popup(' '.join(['Changed', chosen_param, 'from', str(old_value), 'to', str(new_value.get())]))
        else: # set fixed only within range if limits exists
            # lambda apply ifelse condition: new value if time falls in limits range (inclusive), else keep value of chosen_param unchanged
            self.params[chosen_param] = self.params.apply(lambda row: new_value.get() if ((limits[0] <= row['_msec_']) & (row['_msec_'] <= limits[1])) else row[chosen_param], axis = 1)
            self.__update_df()
            self.__popup(' '.join(['Changed', chosen_param, 'to', str(new_value.get()), 'in times', str(limits)]))

    def __slope_choice_window(self, param):
        # choice window
        choose_slope_win = tk.Toplevel()
        choose_slope_win.wm_title('Set time and value for start and end: ' + param)
        choose_slope_win.geometry(self.popup_size)

        t1_time = tk.Entry(choose_slope_win)
        t1_label = tk.Label(choose_slope_win, text = 'start (ms):')
        t1_value = tk.Entry(choose_slope_win)
        t1_label_value = tk.Label(choose_slope_win, text = 'start value')

        t2_time = tk.Entry(choose_slope_win)
        t2_label = tk.Label(choose_slope_win, text = 'end (ms):')
        t2_value = tk.Entry(choose_slope_win)
        t2_label_value = tk.Label(choose_slope_win, text = 'end value')

        variance = tk.Entry(choose_slope_win)
        variance.insert(tk.END, '0') # default value
        variance_label = tk.Label(choose_slope_win, text = 'variance to add')

        new_values = tk.BooleanVar()
        confirm = tk.Button(choose_slope_win, text = 'Confirm', command = lambda: self.__validate_slope(choose_slope_win, new_values, t1_time, t1_value, t2_time, t2_value, variance))

        # lay out elements
        t1_label.grid(row=0,column=0)
        t1_time.grid(row=0,column=1)
        t1_label_value.grid(row=1,column=0)
        t1_value.grid(row=1,column=1)
        
        t2_label.grid(row=3,column=0)
        t2_time.grid(row=3,column=1)
        t2_label_value.grid(row=4,column=0)
        t2_value.grid(row=4,column=1)

        variance_label.grid(row=6,column=0)
        variance.grid(row=6,column=1)

        confirm.grid(row=8,column=0)

        # wait for confirmation before progressing
        self.root.wait_variable(new_values)

    def __validate_slope(self, window, bool_var, t1_time_var, t1_val_var, t2_time_var, t2_val_var, variance_var):
        # to-do: validation of int-status and fitting inside acceptable ranges
        self.__slope_params['t1']['time'] = float(t1_time_var.get())
        self.__slope_params['t1']['value'] = float(t1_val_var.get())
        self.__slope_params['t2']['time'] = float(t2_time_var.get())
        self.__slope_params['t2']['value'] = float(t2_val_var.get())
        self.__slope_params['var'] = float(variance_var.get())
        bool_var.set(True) # mark that we've updated, to allow window to progress
        window.destroy()

    def __validate_int(self, window, stringvar, intvar):
        # to-do: validation of int-status and fitting inside acceptable ranges
        intvar.set(stringvar.get())
        window.destroy()

    def __new_synthesis(self):
        # check if the parameters have changed at all since the original wavdata
        if self.params_wav_unsyncced == False:
            self.__popup("The KLP file hasn't changed.\n\nMake some updates then try again.")
        else: 
            # sanity check
            self.__update_df()
            # initialize synthesizer, add parameters, and run synthesis
            synth = klatt_wrap.synthesizer()
            synth.set_params(self.__convert_params_for_klatt_wrap())
            (new_wavdata,new_rate) = synth.synthesize()
            # save in tmp files
            tmp_wav_path = os.path.join(os.getcwd(),'tmp_synth.wav')
            tmp_klp_path = os.path.join(os.getcwd(),'tmp_synth.klp')
            wavfile.write(tmp_wav_path, new_rate, new_wavdata)
            klpfile.write(tmp_klp_path, synth=synth)
            # add wav and klp data and reset tracking variable
            self.__process_wav_klp(tmp_wav_path, tmp_klp_path)
            self.__add_tg_labels()
            self.params_wav_unsyncced = False
            self.__update_frames()

    def __convert_params_for_klatt_wrap(self):
        # wrapper for visibility of logic
        def apply_conditions(col):
            # check to see if col is a valid klatt parameter
            if (col in klatt_wrap.params_map.keys()) or (col in klatt_wrap.extra_params):
                return True
        # map fixed to int and variable to list
        klatt_params = {col: self.params[col].values.tolist() if self.params[col].nunique() > 1 else self.params[col].unique()[0] for col in self.params.columns if apply_conditions(col)}
        return klatt_params
        
    def __set_zeroes(self):
        for param in self.param_types.keys():
            if self.param_types[param] == 'variable':
                # set variable parameters to 0 in regions without textgrid labels
                self.params[param] = np.where(self.params['label'] == '', 0, self.params[param])
        self.__update_df()

    def __move_to_plot(self):
        # update topframe
        self.topframe = self.plot_frame
        # clear plot information
        for frame in self.plot_frame.winfo_children():
            frame.destroy()
        self.plot_frame.pack_forget()
        plt.close()
        # add in plot info again
        self.plot_frame.pack(fill='both', expand=1)
        self.opts_frame.pack_forget()
        self.__klp_plot_frame()
        
    def __move_to_opts(self):
        # update topframe
        self.topframe = self.opts_frame
        # clear plot information
        for frame in self.plot_frame.winfo_children():
            frame.destroy()
        self.plot_frame.pack_forget()
        plt.close()
        # pack self.opts_frame
        self.opts_frame.pack(side=tk.TOP, fill='both', expand=1)      
        
    def __popup(self, message:str, progress = False):
        self.popup = tk.Toplevel()
        self.popup.geometry(self.popup_size)
        tk.Message(self.popup, text = message, width = 200).pack(side=tk.TOP,fill='both', expand=1)
        if progress is False:
            tk.Button(self.popup, text = 'Got it!', command = self.popup.destroy).pack(side=tk.BOTTOM,fill=tk.X, expand=1)
        elif progress is True: # don't progress until we press continue
            progress_var = tk.BooleanVar(self.popup)
            tk.Radiobutton(self.popup, text = 'Continue...', variable = progress_var, command = self.popup.destroy).pack(side=tk.BOTTOM,fill=tk.X, expand=1)
            return progress_var

    def inSelection(self, point, extents):
        x, y = point
        xmin, xmax, ymin, ymax = extents
        # define in_x
        if (x >= xmin) & (x <= xmax):
            in_x = True
        else:
            in_x = False

        # define in_y
        if (y >= ymin) & (y <= ymax):
            in_y = True
        else:
            in_y = False
            
        # return boolean
        if in_x and in_y:
            return True
        else:
            return False

    def process_wav(self,wavpath):
        # read in wav file
        sr, wav_data = wavfile.read(wavpath)
        wav_times = [1000*(n/sr) for n in range(len(wav_data))]
        return sr, wav_data, wav_times

    def output_params(self):
        return self.params

    def __add_tg_labels(self):
        # select tg labels corresponding to the row where df._msec_ is between self.tg.t1 and .t2.
        for df in [self.params, self.base_params]:
            df['label'] = df.apply(lambda row: self.tg[(self.tg['t1'] <= row['_msec_']) & (self.tg['t2'] >= row['_msec_'])]['label'].to_string(index=False), axis = 1)        

    def make_colormap(self,n,name):
        cmap = plt.get_cmap(name)
        return cmap(np.linspace(0,1,n))
    
    def play(self):
        pg.mixer.quit() # closing if previous mixer was opened; this is to account for a bug where mixer won't override sr if it was previously set
        pg.mixer.pre_init(frequency=self.sr,channels=self.channels)
        pg.mixer.init()
        s = pg.mixer.Sound(buffer=self.wdata)
        s.play()

    def __toggle_selector(self):
        if self.current_toggle is None:
            self.current_toggle = 'selector'
            self.RS.set_active(True)
            self.plot_buttons['Toggle Selector'].configure(background=self.toggled_color, activebackground=self.toggled_color_hover)
        elif self.current_toggle == 'selector': # deactivate
            self.current_toggle = None
            self.RS.set_active(False)
            self.RS.to_draw.set_visible(False)
            self.RS.update()
            # currently storing in dict where key is button label; this is easily broken... to-do: improve                
            self.plot_buttons['Toggle Selector'].configure(background=self.nottoggled_color, activebackground=self.nottoggled_color_hover)
        else:
            self.__popup('You must deactivate any other enabled toggles first.')

    def __toggle_y_drag(self):
        if self.current_toggle is None:
                self.current_toggle = 'y_drag'
                self.plot_buttons['Toggle Y-Dragging'].configure(background=self.toggled_color, activebackground=self.toggled_color_hover)
        elif self.current_toggle == 'y_drag':
            self.current_toggle = None
            # currently storing in dict where key is button label; this is easily broken... to-do: improve
            self.plot_buttons['Toggle Y-Dragging'].configure(background=self.nottoggled_color, activebackground=self.nottoggled_color_hover) 
        else:
            self.__popup('You must deactivate any other enabled toggles first.')

    def __toggle_label_select(self):
        if self.current_toggle is None:
            self.current_toggle = 'label_select'
            self.plot_buttons['Toggle label select'].configure(background=self.toggled_color, activebackground=self.toggled_color_hover)
        elif self.current_toggle == 'label_select':
            self.current_toggle = None
            #currently storing in dict where key is button label; this is easily broken... to-do: improve
            self.plot_buttons['Toggle label select'].configure(background=self.nottoggled_color, activebackground=self.nottoggled_color_hover) 
        else:
            self.__popup('You must deactivate any other enabled toggles first.')

    def __line_select_callback(self, eclick, erelease):
        # clear old selected points if relevant
        try: 
            self.sel_points.set_visible(False)
        except AttributeError: # sel_points not yet instantiated
            pass
        # gather rectangle position
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

    def __on_click(self, event):
        if event.button is MouseButton.RIGHT and self.current_toggle == 'selector':
            self.__rect_pick_points(event)
        elif event.button is MouseButton.LEFT and self.current_toggle == 'y_drag':
            nearest_point, nearest_point_artist = self.__find_neighbor(event)
            if nearest_point is not None:
                self.plot_ax.autoscale(enable=False) # to avoid resizing point shifting plot limits
                self.dragging_point = nearest_point
                self.dragging_point_artist = nearest_point_artist
                self.dragging_point_artist_data = self.dragging_point_artist.get_offsets() 
                # np.where() approach involves masked arrays that are behaving in unexpected ways
                # defaulting to for loop approach
                for i,(x,y) in enumerate(self.dragging_point_artist_data):
                    if (x,y) == self.dragging_point:
                        self.dragging_point_ix = i
                        break
                self.dragging_size_original = self.dragging_point_artist.get_sizes()
        elif event.button is MouseButton.LEFT and self.current_toggle == 'label_select':
            xlims = self.plot_ax.get_xlim()
            ylims = self.plot_ax.get_ylim()
            lim_coords_data = [point for point in zip(xlims,ylims)]
            lim_coords_disp = self.plot_ax.transData.transform(lim_coords_data)
            # get the specified label and let user choose which parameter to change in that label
            curr_label = self.__detect_label(event)
            curr_param = self.__param_choice_window('plotted')
            self.__value_choice_window(curr_param,(curr_label.t1,curr_label.t2))
            self.current_toggle = None # reset toggle logic

    def __on_motion(self, event):
        if self.current_toggle == 'y_drag' and self.dragging_point is not None:
            # get mouse position display coordinates
            mouse_x_data, mouse_y_data = self.plot_ax.transData.inverted().transform((event.x, event.y))
            # overwrite point in artist corresponding to self.dragging_point with new mouse y
            self.dragging_point_artist_data[self.dragging_point_ix] = (self.dragging_point[0], mouse_y_data)
            self.dragging_point = (self.dragging_point[0], mouse_y_data)
            self.dragging_point_artist.set_offsets(self.dragging_point_artist_data)
            # set size to triple while dragging
            new_sizes = np.repeat(self.dragging_size_original,len(self.dragging_point_artist_data))
            new_sizes[self.dragging_point_ix] = 3*self.dragging_size_original[0]
            self.dragging_point_artist.set_sizes(new_sizes)
            self.canvas.draw()

    def __on_release(self, event):
        if event.button is MouseButton.LEFT and self.current_toggle == 'y_drag':
            if self.dragging_point is not None:
                # get param that self.dragging_point_artist corresponds to (find key matching value)
                param = list(self.data_artists.keys())[list(self.data_artists.values()).index(self.dragging_point_artist)]
                # update self.params
                row_index = self.params.loc[self.params['_msec_'] == self.dragging_point[0]].index
                self.params.loc[row_index, param] = self.dragging_point[1]
                self.__update_df(update_frames=False)
                # reset sizes and draw updated canvas
                self.dragging_point_artist.set_sizes(self.dragging_size_original)
                self.canvas.draw()
            self.dragging_point = None
            self.dragging_point_artist = None
            self.dragging_point_artist_data = None
            self.dragging_point_ix = None
            self.dragging_size_original = None

    def __detect_label(self, event): # to-do: see if this is useful: https://matplotlib.org/stable/_modules/matplotlib/backend_bases.html#FigureCanvasBase.inaxes
        # convert from display coordinates of event to data coordinates
        mouse_x_data, mouse_y_data = self.plot_ax.transData.inverted().transform((event.x, event.y))
        for row in self.tg.itertuples():
            if row.label != '':
                # detect if mouse click is in label interval
                if row.t1 <= mouse_x_data <= row.t2:
                    return row # row is named tuple with properties: Index, t1, t2, label, and fname

    def __find_neighbor(self, event):
        """
        This method and draggable approach inspired largely by https://github.com/yuma-m/matplotlib-draggable-plot
        """
        xlims = self.plot_ax.get_xlim()
        ylims = self.plot_ax.get_ylim()
        lim_coords_data = [point for point in zip(xlims,ylims)]
        lim_coords_disp = self.plot_ax.transData.transform(lim_coords_data)
        ax_disp_width = lim_coords_disp[1][0]-lim_coords_disp[0][0]
        ax_disp_height = lim_coords_disp[1][1]-lim_coords_disp[0][1]

        distance_threshold = max([ax_disp_height,ax_disp_width])/100
        min_distance = 2*distance_threshold

        nearest_point = None
        nearest_point_artist = None

        for artist in self.data_artists.values():
            points = artist.get_offsets()
            for [x,y] in points:
                # convert data x,y into display coords
                display_x, display_y = self.plot_ax.transData.transform((x, y))
                distance = np.hypot(event.x - display_x, event.y - display_y)
                if distance < min_distance:
                    min_distance = distance
                    nearest_point = (x, y)
                    nearest_point_artist = artist
        if min_distance < distance_threshold:
            return nearest_point, nearest_point_artist
        else:
            return None, None

    def __rect_pick_points(self, event):   
        # get coordinates of current selector
        x1, x2, y1, y2 = self.RS.extents
        
        sub_xs = []
        sub_ys = []
        
        # loop over all data artists in self.plot_ax
        for artist in self.data_artists.values():
            points = artist.get_offsets()
            for [x,y] in points:
                if (x1 < x < x2) and (y1 < y < y2):
                    sub_xs.append(x)
                    sub_ys.append(y)
        self.plot_ax.autoscale(enable=False) # ensuring new scatter doesn't change ylim
        self.sel_points = self.plot_ax.scatter(sub_xs,sub_ys,c='red')
        self.canvas.draw()
    
    def __edit_sel_convert_av(self): # to-do: convert this to a more general "swapping" function
        if hasattr(self,'sel_points'):
            xmin = min([pair[0] for pair in self.sel_points.get_offsets()])
            xmax = max([pair[0] for pair in self.sel_points.get_offsets()])
            
            for index, row in self.params.iterrows():
                # find self.params rows in selected range
                if self.inSelection((row['_msec_'],row['af']), self.RS.extents):
                    if row['af'] > row['av']:
                        # swap values
                        self.params.loc[index, 'av'] = row['af']
                        self.params.loc[index, 'af'] = row['av']
            self.__update_df()
        else:
            self.__popup('Must select points before calling this function.')
    
    def __interpolate_outliers(self):
        if hasattr(self,'sel_points'):
            xmin = min([pair[0] for pair in self.sel_points.get_offsets()])
            xmax = max([pair[0] for pair in self.sel_points.get_offsets()])
            
            matches = []
            # find all variable cells that fall in rect selection
            for index, row in self.params.iterrows():
                for col in self.params.columns:
                    if col not in ['_msec_', 'label']:
                        if self.inSelection((row['_msec_'],row[col]), self.RS.extents):
                            matches.append((index,col))
            
            # get all unique columns matched in rectangular selection
            cols = list(set([col for ix, col in matches]))
            if cols: # there are columns in set
                chosen_col = self.__param_choice_window('specified',cols)

            # getting row and column indices for selection
            chosen_indices = [ix for ix, col in matches if col == chosen_col]
            chosen_col_ix = self.params.columns.get_loc(chosen_col)
            chosen_col_ixs_both = [self.params.columns.get_loc(c) for c in ['_msec_',chosen_col]]
            # this method appears to copy; changes won't affect underlying structure
            sub = self.params.iloc[chosen_indices,chosen_col_ixs_both]
        
            # calculate linear regression
            regr = LinearRegression() # from sklearn.linear_model
            xdata = sub['_msec_'].values.reshape(-1,1)
            ydata = sub[chosen_col].values.reshape(-1,1)
            regr.fit(xdata, ydata)
            
            self.plot_ax.plot(xdata.reshape(-1),regr.predict(xdata).reshape(-1), color = 'red')
            self.canvas.draw()
            
            # detect outlier residuals
            resid = {index: ydata.reshape(-1)[index] - regr.predict([[x]]).reshape(-1) for index, x in enumerate(xdata.reshape(-1))}
            mean = np.mean(list(resid.values()))
            sd = np.std(list(resid.values()))
            ll = mean - self.outlier_threshold * sd
            ul = mean + self.outlier_threshold * sd
            
            # accept and exclude based on sd threshold
            # gather indices (defined in terms of position in xdata, not sub!)
            accepted_ixs = [ix for ix in resid.keys() if  ll <= resid[ix] <= ul] 
            outlier_ixs = [ix for ix in resid.keys() if not (ll <= resid[ix] <= ul)]
            
            if outlier_ixs:
                # new linear fit on accepted values only
                xnew = np.array(xdata)[accepted_ixs].reshape(-1,1)
                ynew = np.array(ydata)[accepted_ixs].reshape(-1,1)
                regr_new = LinearRegression() # from sklearn.linear_model
                regr_new.fit(xnew,ynew)
                
                # recover _ms_ and new predictions for outliers
                pred_msec_ = xdata[outlier_ixs].reshape(-1)
                pred_new = regr_new.predict(np.array(xdata)[outlier_ixs]).reshape(-1)
                
                # 
                for ms, pred in zip(pred_msec_,pred_new):
                    # determine sub row index matching prediction _msec_
                    pos = sub[sub['_msec_'] == ms].index.item()
                    # replace old sub value with new prediction
                    sub.loc[pos,chosen_col] = pred
            else:
                self.__popup('no outliers found!')
                
            # to-do: dialog box to preview changes and approve or deny
            
            # update self.params and merge back together
            self.params.iloc[chosen_indices,chosen_col_ix] = sub[chosen_col]
        
            # to-do: retire this approach once the preview dialog is implemented
            # be sure to uncomment the regular self.__update_df
            self.root.after(2000, self.__update_df)
            #self.__update_df()
        else: # no points selected
            self.__popup('Must select points before calling this function.')

    def __adjust_near_zeroes(self,threshold_percent = 0.08):
        any_adjusted = False
        # loop over all data artists in self.plot_ax
        for param, artist in self.data_artists.items():
            points = artist.get_offsets()
            max_y_val = max([y for (x, y) in points])
            # set y threshold under which we adjust to 0
            threshold = max_y_val * threshold_percent
            # also adjust any negative parameters
            to_adjust = [(x,y) for (x,y) in points if (float("-inf") < y < 0) or (0 < y <= threshold)]
            if to_adjust:
                any_adjusted = True
                for (msec, val) in to_adjust:
                    row_index = self.params[self.params['_msec_'] == msec].index.item()
                    self.params.loc[row_index,param] = 0
                self.__popup('Zeroed ' + str(len(to_adjust)) + ' point(s) in axis ' + param)

        if any_adjusted is True:
            self.__update_df()
        else:
            self.__popup('No adjustments made to current parameters.')

    def __draw_slope(self):
        # initialize slope info dict
        self.__slope_params = {'t1': {}, 't2': {}, 'param': None, 'var': 0}
        # get param to plot and start/end information
        curr_param = self.__param_choice_window('plotted')
        self.__slope_params['param'] = curr_param
        self.__slope_choice_window(curr_param)
        t1 = self.__slope_params['t1']['time']
        t2 = self.__slope_params['t2']['time']
        t1_val = self.__slope_params['t1']['value']
        t2_val = self.__slope_params['t2']['value']

        # check that start and end times are within current range; otherwise return function early
        if (t1 <= self.params['_msec_'].min()) or (t2 >= self.params['_msec_'].max()):
            self.__popup('Start and end times must be within plotted range!')
            return

        # find param data points within given slope start/stop
        idx = self.params.index[(t1 <= self.params['_msec_']) &  (self.params['_msec_'] <= t2)]
         
        # calculate new values based on input slope
        new_values = [((((t2_val-t1_val)/(len(idx)-1)) * n) + t1_val) for n in range(len(idx))]

        # calculate variances and add to new values
        variances = np.random.normal(size=len(new_values), scale=self.__slope_params['var'])
        new_values = variances + new_values

        # overwrite
        self.params.loc[idx,curr_param] = new_values

        # reset slope_params
        self.__slope_params = None
        self.__update_df()

    def __make_plot_buttons(self):
        # frame to contain plot-related buttons
        plot_buttons_container = tk.Frame(self.plot_frame, borderwidth=5, background='#8093F1', relief=tk.GROOVE)
        # establish buttons
        b0 = tk.Button(plot_buttons_container, text = 'Toggle Selector', command = self.__toggle_selector)
        b1 = tk.Button(plot_buttons_container, text = 'Toggle Y-Dragging', command = self.__toggle_y_drag)
        b2 = tk.Button(plot_buttons_container, text = 'Toggle label select', command = self.__toggle_label_select)
        b3 = tk.Button(plot_buttons_container, text = 'Zero non-labels', command = self.__set_zeroes)
        b4 = tk.Button(plot_buttons_container, text = 'af/av correction sel.', command = self.__edit_sel_convert_av)
        b5 = tk.Button(plot_buttons_container, text = 'Interpolate sel.', command = self.__interpolate_outliers)
        b6 = tk.Button(plot_buttons_container, text = 'Adjust near zeroes', command = self.__adjust_near_zeroes)
        b7 = tk.Button(plot_buttons_container, text = 'Draw Slope', command = self.__draw_slope)

        # layout and pack buttons
        self.plot_buttons = {button['text']: button for button in [b0,b1,b2,b3,b4,b5,b6,b7]}
        ncols = round(len(self.plot_buttons)/2)
        self.__neat_buttons(plot_buttons_container, ncols, self.plot_buttons)
        plot_buttons_container.pack(side=tk.BOTTOM, fill=tk.X)

    def __klp_plot_frame(self):
        # track currently selected parameters to plot
        self.selected_params = {v: isSelected for v, isSelected in self.param_booleans.items() if isSelected.get() is True}
        # container for passing references to the artist collections
        self.data_artists = {}
        # generate selected colormap for parameters
        n_selected = sum([v.get() for v in self.selected_params.values()])
        selected_cmap = {param:self.make_colormap(n_selected,'tab20b')[index] for index, param in enumerate(self.selected_params.keys())}

        # set up stacked plots
        fig, (ax1, self.plot_ax) = plt.subplots(2,1,figsize = (8,6), sharex=True)
        fig.subplots_adjust(hspace=0.5)
        self.plot_ax.set_title('When selector is toggled, right click to highlight selected points.',
                     y=(-0.2*self.plot_ax.get_ylim()[1]), verticalalignment='top')
        
        # visualize wave file
        ax1.plot(self.wtimes,self.wdata)
        
        # if wdata is out of date to updated klp params, add warning
        if self.params_wav_unsyncced == True:
            midy = ax1.get_ylim()[0] + ((ax1.get_ylim()[1] - ax1.get_ylim()[0])/2)
            ax1.text(ax1.get_xlim()[1],midy,'WARNING:\nwav\noutdated!',color='red')
            
        # visualize variable parameters
        for param in self.selected_params.keys():
            # plot variable parameters
            data_artist = self.plot_ax.scatter(self.params['_msec_'],self.params[param],color=selected_cmap[param])
            self.plot_ax.text(1.05*self.plot_ax.get_xlim()[1], np.mean(self.params[param]), param, fontsize=14, color=selected_cmap[param])
            # tracking which parameters are active and their artist collection
            self.data_artists[param] = data_artist

        # get array of non-empty self.tg labels
        filtered_labels = self.tg[self.tg.label != ''].label.values
        # assign self.tg label colormap
        n_color = len(filtered_labels)
        tg_cmap = {label: self.make_colormap(n_color,'Pastel1')[index] for index, label in enumerate(filtered_labels)}
        # plot self.tg label overlays
        for index, r in self.tg.iterrows():
            if r['label'] != '':
                midx = r['t1'] + ((r['t2'] - r['t1'])/2)
                # color overlays
                ax1.axvspan(r['t1'], r['t2'], alpha=0.5, color=tg_cmap[r['label']])
                self.plot_ax.axvspan(r['t1'], r['t2'], alpha=0.5, color=tg_cmap[r['label']])
                # add labels above overlay # to-do: these are expressed in terms of data coords; consider plot xform instead? current instantiation is buggy
                ax1.text(midx, 1.05*ax1.get_ylim()[1], r['label'], fontsize=14)
                self.plot_ax.text(midx, 1.05*self.plot_ax.get_ylim()[1], r['label'], fontsize=14)
        
        # creating the Tkinter canvas containing the Matplotlib figure
        self.canvas = FigureCanvasTkAgg(fig, master = self.plot_frame)  
        self.canvas.draw()

        # creating the Matplotlib toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        toolbar.update()

        # placing the toolbar on the Tkinter window
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # placing the plot button row
        self.__make_plot_buttons()

        # establish the RectangleSelector box
        self.RS = RectangleSelector(self.plot_ax, self.__line_select_callback,
            drawtype='box', useblit=False, button=[1], 
            minspanx=5, minspany=5, spancoords='pixels', 
            interactive=True)
        self.RS.set_active(False)

        # connect the canvas to relevant key and mouse events
        # to-do: check after development if all of these connections actually needs the canvas lambda-ed into them, or if a normal function pass is fine.
        self.canvas.mpl_connect('button_press_event', self.__on_click) #lambda event: self.__on_click(event, canvas))
        self.canvas.mpl_connect('button_release_event', self.__on_release) #lambda event: self.__on_release(event, canvas)) 
        self.canvas.mpl_connect('motion_notify_event', self.__on_motion)#lambda event: self.__on_motion(event, canvas))
    