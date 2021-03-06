# klsyngui - A Gui for the Klatt Synthesizer

## About this project

This is a GUI that provides additional functionality for Ron Sprouse's klsyn package (https://github.com/rsprouse/klsyn). The klsyn package is a Python port of Dennis Klatt's original C speech synthesis software. Please be sure to review the licensing information in klsyn's README for restrictions on using that package.



## Dependencies
Development was originally carried out in Python 3.8.5

- https://github.com/rsprouse/klsyn
- https://github.com/rsprouse/audiolabel
- pandas (developed using 1.2.4)
- pandastable (developed using 0.12.2.post1)
- tk/tkinter (developed using 0.1.0)
- matplotlib (developed using 3.3.2)
- sklearn (developed using 0.23.2)
- scipy (developed using 1.5.2)
- pygame (developed using 2.0.2)

## Installation

## Usage

## Task List

- [x] Initial Development
- [ ] Installation blurb
- [ ] Usage blurb
- [ ] custom_read() functionality in pull request to klsyn.klpfile.read()
- [ ] Check against infinite hanging in self.\_\_check\_parameter\_fixedvariable\_status() and self.\_\_param\_choice\_window()
- [ ] Implement selection window in self.\_\_check\_parameter\_fixedvariable\_status()
- [ ] Validation of int status and ranges in self.\_\_validate\_slope() and self.\_\_validate\_int()
- [ ] Improve hard coded reference to button labels in various self.\_\_toggle functions
- [ ] Convert self.\_\_edit\_sel\_convert\_av() into general swapping function
- [ ] Add confirmation window to self.\_\_interpolate\_outliers()
- [ ] Improve location of plot labels in self.\_\_klp\_plot\_frame()
- [ ] Add setup.py and related files for pip install


## Contact
If you have questions about using this module, please contact me at wilbanks.ericw@gmail.com.
