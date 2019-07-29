import numpy as np
import scipy.stats as stats
from numpy.testing import assert_array_equal, assert_allclose

from optbayesexpt import ProbDistFunc

myPDF = ProbDistFunc()

p1 = np.arange(4)
p2 = np.arange(3)
params = (p1, p2)

"""test pdf_config()"""
myPDF.pdf_config(params)
def test_pdf_paramvals():
    [assert_array_equal(np.array(i), np.array(j)) for i, j in zip(myPDF.paramvals, params)]

def test_pdf_shape():
    assert_array_equal(myPDF.pdfshape, np.array((4,3)))

"""test set_pdf methods"""
def test_set_pdf_flat():
    zero_result = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
    one_result = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]])
    myPDF.set_pdf(flat=True)
    assert_array_equal(myPDF.lnPDF, zero_result)
    assert_array_equal(myPDF.PDF, one_result)

px = [.1, .2, .3, .1]
py = [.1, 1, .1]
expected_unnormalized_pdf = np.array([[.01, .1, .01],
                                      [.02, .2, .02],
                                      [.03, .3, .03],
                                      [.01, .1, .01]])
expected_unnormalized_lnpdf = np.log(expected_unnormalized_pdf)

def test_set_pdf_probvalarrays():
    myPDF.set_pdf(probvalarrays = [px, py])
    assert_allclose(myPDF.PDF, expected_unnormalized_pdf)


def test_set_pdf_pdf():
    myPDF.set_pdf(pdf=expected_unnormalized_pdf)
    assert_allclose(myPDF.PDF, expected_unnormalized_pdf)
    assert_allclose(myPDF.lnPDF, expected_unnormalized_lnpdf)


def test_set_pdf_lnpdf():
    myPDF.set_pdf(lnpdf=expected_unnormalized_lnpdf)
    assert_allclose(myPDF.PDF, expected_unnormalized_pdf)
    assert_allclose(myPDF.lnPDF, expected_unnormalized_lnpdf)


"""test PDF updates"""

def test_add_lnpdf():
    myPDF.set_pdf(flat=True)
    # PDF = ones, lnpdf = zeros
    myPDF.add_lnpdf(np.ones(myPDF.pdfshape))
    assert_allclose(myPDF.PDF, np.ones((4,3)))
    assert_allclose(myPDF.lnPDF, np.zeros((4,3)))


def test_multiply_pdf():
    myPDF.set_pdf(flat=True)
    myPDF.multiply_pdf(np.ones(myPDF.pdfshape))
    assert_allclose(myPDF.PDF, np.ones((4, 3)))
    assert_allclose(myPDF.lnPDF, np.zeros((4,3)))


"""test pdf output methods"""

def test_max_params():
    myPDF.set_pdf(pdf=expected_unnormalized_pdf)
    maxpars = myPDF.max_params()
    assert_allclose(maxpars, (2, 1))


def test_get_pdf():
    myPDF.set_pdf(flat=True)
    assert_array_equal(myPDF.get_pdf(), np.ones((4,3))/12.)
    assert_array_equal(myPDF.get_pdf(denuisance=(0,)), np.ones(3)/3.)
    assert_array_equal(myPDF.get_pdf(denuisance=(1,)), np.ones(4)/4.)

def test_get_std():
    myPDF.set_pdf(flat=True)
    assert_allclose(myPDF.get_std(0), 1.118034, rtol=1e-6)
    assert_allclose(myPDF.get_std(1), 0.816497, rtol=1e-6)

def test_get_mean():
    myPDF.set_pdf(flat=True)
    assert_allclose(myPDF.get_mean(0), (1.5, 1.118034), rtol=1e-6)
    assert_allclose(myPDF.get_mean(1), (1.0, 0.816497), rtol=1e-6)

def test_entropy():
    myPDF.set_pdf(pdf=expected_unnormalized_pdf)
    testpdf = expected_unnormalized_pdf/expected_unnormalized_pdf.sum()
    assert_allclose(myPDF.entropy(), stats.entropy(testpdf.flatten(), base=2), rtol=1e-6)