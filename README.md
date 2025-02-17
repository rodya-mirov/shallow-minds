# shallow-minds: A short course on artifical neural networks.

This will be taught through a series of Jupyter notebooks.  Relevant code will be captured in python modules (currently just `basic_nn.py`), and these modules will be sufficient to compare the effectiveness of various techniques.

These modules are *not* intended to replace modern, performant libraries such as Theano, Torch, or TensorFlow; instead, they are intended to educate the reader on the basic techniques and start-of-the-art variations.  They are also intended for the reader to be able to experiment with the code in a hassle-free manner with as low an entry barrier as possible.

Notebooks are numbered, and the reader is encouraged to read them in order, as notebooks may reference code or reasoning from an earlier notebook.  As soon as this is possible (the notebook titled "A First Simulation") there will be enough moving parts for the reader to experiment on his/her own and discover what the advantages and disadvantages of different choices are.

### Required Technologies
You'll need to get `python 3`, `numpy`, and `jupyter notebook`.  Any installation of `anaconda` with python 3 will cover everything you need. Python 2 users are encouraged to upgrade, although the code should be easy enough to downgrade by hand. It is assumed that the reader can open and interact with jupyter notebooks; if not, Jupyter has an excellent <a href="https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/">guide to getting started.</a>

### Getting Started

If you've already installed anaconda, you're basically done. There are two ways to go from here.

#### Non-interactive way:
1. Read the notebooks' converted forms in your browser; Github will handle this automatically.
2. Download the code files and play with them as your interest dictates.

#### Interactive way:
1. Download the notebooks' original forms (maybe clone the repo)
2. Run Jupyter locally on your machine so that you can access the notebooks.
3. Run each notebook individually.
4. At the end of each notebook, run your own experiments with what we've done so far, and see if the results match your intuition.  Build up your understanding by guess and check!

### Want to help out?

I'd love suggestions for what to cover next, from techniques to cover or papers to integrate. If some phrasing is unclear or wrong, let me know; I'd like to fix it.

I'd also like suggestions for improving the performance of the training functions.  However, such suggestions *should not impact the readability or simplicity of the functions* - remember this is about learning, not being enterprise-grade software. This is not intended to replace existing libraries!

### Questions / Issues
B1. *There are little lines next to all the math!*

A1. This is a problem happening between older Mathjax installations and Chrome. You can force your Jupyter installation to use current mathjax and fix it locally, but it will always look bad on Github until they update theirs.  Or you can use Firefox (or etc.).

B2. *The explanation for back-propagation is confusing?*

A2. I know.  I'm sorry.  It's a hard topic to explain.  Also, I think there are some notational problems in the explanation, which only make it harder.  I'm going to go back and fix it soon.  The code works though, so if you got the gist, just copy the code and move on.

### What's Next?
1. More and bigger datasets
2. Dropout (a new kind of regularization which is sort of an ensemble method)
3. Harder ideas which go beyond gradient descent and its tweaks (e.g. unsupervised training by level, pooling, and so on).
