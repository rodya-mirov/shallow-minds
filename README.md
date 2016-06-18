# shallow-minds: A short course on artifical neural networks.

This will be taught through a series of Jupyter notebooks.  Relevant code will be captured in python modules (currently just `basic_nn.py`), and these modules will be sufficient to compare the effectiveness of various techniques.

These modules are *not* intended to replace modern, performant libraries such as Theano, Torch, or TensorFlow; instead, they are intended to educate the reader on the basic techniques and start-of-the-art variations.  They are also intended for the reader to be able to experiment with the code in a hassle-free manner with as low an entry barrier as possible.

Notebooks are numbered, and the reader is encouraged to read them in order, as notebooks may reference code or reasoning from an earlier notebook.  As soon as this is possible (the notebook titled "A First Simulation") there will be enough moving parts for the reader to experiment on his/her own and discover what the advantages and disadvantages of different choices are.

**Required technologies:** `python 3`, `numpy`, and `jupyter notebook`.  Any installation of `anaconda` with python 3 will cover everything you need. Python 2 users are encouraged to upgrade, although the code should be easy enough to downgrade by hand. It is assumed that the reader can open and interact with jupyter notebooks; if not, Jupyter has an excellent <a href="https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/">guide to getting started.</a>

**Getting started:** If you've already installed anaconda, you're basically done. There are two ways to go from here.

Non-interactive way:
1. Read the notebooks' converted forms in your browser.
2. Download the code files and play with them as your interest dictates.

Interactive way:
1. Download the notebooks' original forms (maybe clone the repo)
2. Run Jupyter locally on your machine so that you can access the notebooks.
3. Run each notebook individually.
4. At the end of each notebook, run your own experiments with what we've done so far, and see if the results match your intuition.  Build up your understanding by guess and check!

**Want to help out?** I'd love suggestions for what to cover next, from techniques to cover or papers to integrate. If some phrasing is unclear or wrong, let me know; I'd like to fix it.

I'd also like suggestions for improving the performance of the training functions.  However, such suggestions *should not impact the readability or simplicity of the functions* - remember this is about learning, not being enterprise-grade software. This is not intended to replace existing libraries!