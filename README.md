

<div align="center">
  <img src="https://raw.githubusercontent.com/chandraveshchaudhari/personal-information/initial_setup/logos/my%20github%20logo%20template-HSFSI%20framework.drawio.png" width="640" height="320">
</div>

# Hybrid Subset Feature Selection and Importance Framework


- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Contribution](#contribution)
- [Future Improvements](#future-improvements)

## Introduction
The feature selection algorithms MultiSURF (Urbanowicz et al., 2017), MultiSURF*
(Granizo-Mackenzie & Moore, 2013), ReliefF (Kononenko et al., 1994), TuRF (Moore
& White, 2007), SURF (Greene et al., 2009), SURF* (Greene et al., 2010) are imple-
mented as described in Urbanowicz et al. (2017) paper. These feature selection al-
gorithms differ in the number of nearest neighbours, algorithms’ computational ef-
ficiency, and scoring methodology for selecting near or far instances. Another ap-
proach of feature selection involves iteratively removing or adding features to con-
struct a feature subset, guided by an estimator. The HSFSI framework provides two
meta-transformers implemented using the scikit-learn library based on the impor-
tance weights of linear support vector classifier with L1 penalty and extra Trees Clas-
sifier with 50 estimators for selecting features

### Authors
<img align="left" width="231.95" height="75" src="https://raw.githubusercontent.com/chandraveshchaudhari/personal-information/initial_setup/images/christ.png">

This work is part of Thesis of [Chandravesh chaudhari][chandravesh linkedin], Doctoral candidate at [CHRIST (Deemed to be University), Bangalore, India][christ university website] under supervision of [Dr. Geetanjali purswani][geetanjali linkedin].

<br/>

[chandravesh linkedin]: https://www.linkedin.com/in/chandravesh-chaudhari "chandravesh linkedin profile"
[geetanjali linkedin]: https://www.linkedin.com/in/dr-geetanjali-purswani-546336b8 "geetanjali linkedin profile"
[christ university website]: https://christuniversity.in/ "website"

## Features
- replicable
- customisable

#### Significance
- Saves time

## Installation 
This project is available at [PyPI](https://pypi.org/project/systematic-reviewpy/). For help in installation check 
[instructions](https://packaging.python.org/tutorials/installing-packages/#installing-from-pypi)
```bash
python3 -m pip install systematic-reviewpy  
```

### Dependencies
##### Required
- [rispy](https://pypi.org/project/rispy/) - A Python 3.6+ reader/writer of RIS reference files.
- [pandas](https://pypi.org/project/pandas/) - A Python package that provides fast, flexible, and expressive data 
structures designed to make working with "relational" or "labeled" data both easy and intuitive.
##### Optional
- [browser-automationpy](https://github.com/chandraveshchaudhari/browser-automationpy/)
- [pdftotext](https://pypi.org/project/pdftotext/) - Simple PDF text extraction
- [PyMuPDF](https://pypi.org/project/PyMuPDF/) - PyMuPDF (current version 1.19.2) - A Python binding with support for 
MuPDF, a lightweight PDF, XPS, and E-book viewer, renderer, and toolkit.

## Important links
- [Documentation](https://chandraveshchaudhari.github.io/systematic-reviewpy/)
- [Quick tour](https://chandraveshchaudhari.github.io/systematic-reviewpy/systematic-reviewpy%20tutorial.html)
- [Project maintainer (feel free to contact)](mailto:chandraveshchaudhari@gmail.com?subject=[GitHub]%20Source%20sytematic-reviewpy) 
- [Future Improvements](https://github.com/chandraveshchaudhari/systematic-reviewpy/projects)
- [License](https://github.com/chandraveshchaudhari/systematic-reviewpy/blob/master/LICENSE.txt)

## Contribution
all kinds of contributions are appreciated.
- [Improving readability of documentation](https://chandraveshchaudhari.github.io/systematic-reviewpy/)
- [Feature Request](https://github.com/chandraveshchaudhari/systematic-reviewpy/issues/new/choose)
- [Reporting bugs](https://github.com/chandraveshchaudhari/systematic-reviewpy/issues/new/choose)
- [Contribute code](https://github.com/chandraveshchaudhari/systematic-reviewpy/compare)
- [Asking questions in discussions](https://github.com/chandraveshchaudhari/systematic-reviewpy/discussions)

## Future Improvements
- [ ] Web based GUI


