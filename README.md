# klsyngui - A Gui for the Klatt Synthesizer

## About this project

This is a GUI that provides additional functionality for Ron Sprouse's klsyn package (https://github.com/rsprouse/klsyn). The klsyn package is a Python port of Dennis Klatt's original C speech synthesis software. Please be sure to review the licensing information in klsyn's README for restrictions on using that package.


## Dependencies
- https://github.com/rsprouse/klsyn
- https://github.com/rsprouse/audiolabel
- pandas
- pandastable
- tkinter
- matplotlib
- sklearn
- scipy
- pygame

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


## Contact
If you have questions about using this module, please contact me at wilbanks.ericw@gmail.com.