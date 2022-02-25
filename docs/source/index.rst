Poisson Factorization
=====================
This is the documentation page for the Python package *poismf*, which produces approximate
non-negative low-rank matrix factorizations of sparse counts matrices by maximizing Poisson
likelihood minus a regularization term, the result of which can be used for e.g. implicit-feedback
recommender systems or bag-of-words-based topic modeling.


For more information, visit the project's GitHub page:

`<https://www.github.com/david-cortes/poismf>`_

For the R version, see the CRAN page:

`<https://cran.r-project.org/package=poismf>`_

Installation
============
The Python version of this package can be easily installed from PyPI
::

   pip install poismf

(See the GitHub page for more details)

Quick Example
=============

* `Poisson Factorization on the LastFM dataset <http://nbviewer.jupyter.org/github/david-cortes/poismf/blob/master/example/example_poismf_lastfm.ipynb>`_.

Methods
=======

* `PoisMF <#poismf.PoisMF>`_
* `fit <#poismf.PoisMF.fit>`_
* `fit_unsafe <#poismf.PoisMF.fit_unsafe>`_
* `predict <#poismf.PoisMF.predict>`_
* `predict_factors <#poismf.PoisMF.predict_factors>`_
* `topN <#poismf.PoisMF.topN>`_
* `topN_new <#poismf.PoisMF.topN_new>`_
* `transform <#poismf.PoisMF.transform>`_

PoisMF
======

.. automodule:: poismf
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
