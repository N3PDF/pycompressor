# This file defines the x-grid and computes the PDF
# gird according to the input parameters.

import lhapdf
import numpy as np


class XGrid:
    """Construct the x-grid as in NNPDF.

    Returns
    -------
        result: array_like
            Numpy array containg the x-grid
    """

    def __init__(self):
        self.x_nodes = [
            1.0974987654930569e-05,
            1.5922827933410941e-05,
            2.3101297000831580e-05,
            3.3516026509388410e-05,
            4.8626015800653536e-05,
            7.0548023107186455e-05,
            1.0235310218990269e-04,
            1.4849682622544667e-04,
            2.1544346900318823e-04,
            3.1257158496882353e-04,
            4.5348785081285824e-04,
            6.5793322465756835e-04,
            9.5454845666183481e-04,
            1.3848863713938717e-03,
            2.0092330025650459e-03,
            2.9150530628251760e-03,
            4.2292428743894986e-03,
            6.1359072734131761e-03,
            8.9021508544503934e-03,
            1.2915496650148829e-02,
            1.8738174228603830e-02,
            2.7185882427329403e-02,
            3.9442060594376556e-02,
            5.7223676593502207e-02,
            8.3021756813197525e-02,
            0.10000000000000001,
            0.11836734693877551,
            0.13673469387755102,
            0.15510204081632653,
            0.17346938775510204,
            0.19183673469387758,
            0.21020408163265308,
            0.22857142857142856,
            0.24693877551020407,
            0.26530612244897961,
            0.28367346938775512,
            0.30204081632653063,
            0.32040816326530613,
            0.33877551020408170,
            0.35714285714285710,
            0.37551020408163271,
            0.39387755102040811,
            0.41224489795918373,
            0.43061224489795924,
            0.44897959183673475,
            0.46734693877551026,
            0.48571428571428565,
            0.50408163265306127,
            0.52244897959183678,
            0.54081632653061229,
            0.55918367346938780,
            0.57755102040816331,
            0.59591836734693870,
            0.61428571428571421,
            0.63265306122448983,
            0.65102040816326534,
            0.66938775510204085,
            0.68775510204081625,
            0.70612244897959175,
            0.72448979591836737,
            0.74285714285714288,
            0.76122448979591839,
            0.77959183673469379,
            0.79795918367346941,
            0.81632653061224492,
            0.83469387755102042,
            0.85306122448979593,
            0.87142857142857133,
            0.88979591836734695,
            0.90816326530612246,
        ]

    def build_xgrid(self):
        """
        Construct an array of input pdf
        """
        x_grid = np.array(self.x_nodes)
        return x_grid


class PdfSet:
    """This formats the input PDF replicas. It returns a multi-
    dimensional array that has the following shape (flavor,
    pdfReplicas, pdfValue).

    Parameters
    ----------
        pdf_name: str
            Name of the input PDF replicas
        xgrid: array
            Array of x-grid
        q_value: float
            Value of intial energy scale Q
        nf: int
            Total number of flavors
    """

    def __init__(self, pdf_name, xgrid, q_value, nf):
        self.nf = nf
        self.xgrid = xgrid
        self.q_value = q_value
        self.pdf_name = pdf_name

    def build_pdf(self):
        """Construct a grid of PDF from a grid in x depenging on
        the input parameters.

        Returns
        -------
            result: array_like
                PDF grid of shape=(replicas, flavours, x-grid)
        """
        # Call lhapdf PDF set PDF set
        pdf = lhapdf.mkPDFs(self.pdf_name)
        pdf_size = len(pdf) - 1
        xgrid_size = self.xgrid.shape[0]

        # construct input pdf replicas
        inpdf = np.zeros((pdf_size, 2 * self.nf + 1, xgrid_size))
        for p in range(pdf_size):
            for f in range(-self.nf, self.nf + 1):
                for x in range(xgrid_size):
                    inpdf[p][f + self.nf][x] = pdf[p + 1].xfxQ(
                        f, self.xgrid[x], self.q_value
                    )
        return inpdf
