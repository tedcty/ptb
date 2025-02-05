import numpy as np


class PolyFourier(object):

    # Function to calculate polynomial plus Fourier coefficients
    # to fit the specified (t,y) input testdata.
    # Inputs: t - indepdendent variable
    #         y - dependent variable
    #         degree - polynomial degree (0 through 7)
    #         nharmonics - number of Fourier harmonics
    # This function assumes that t = tStart to tEnd defines one cycle
    # of periodic or nonperiodic testdata.

    @staticmethod
    def polyFourierCoefs(t,y,degree,nharmonics):

        # Calculate the frequency w from the final time tf.
        tf = np.max(t) - np.min(t)
        w = 2.0 * np.pi / tf

        # Allocate memory for linear least squares matrix and vectors.

        ncoefs = 2 * nharmonics + (degree + 1)
        npts = len(t)  # Use last point, even if slightly nonperiodic
        time = t

        A = np.zeros([npts, ncoefs])
        x = np.zeros([ncoefs, 1])
        b = np.zeros([npts, 1])

        # Fill in the first columns of the A matrix based on the polynomial degree.
        if degree == 0:
            A[:, 0] = np.ones(npts)

        elif degree == 1:
            A[:, 0] = np.ones([1,npts])
            A[:, 1] = time

        elif degree == 2:
            time2 = np.multiply(time, time)

            A[:, 0] = np.ones(npts)
            A[:, 1] = time
            A[:, 2] = time2

        elif degree == 3:
            time2 = np.multiply(time, time)
            time3 = np.multiply(time2, time)

            A[:, 0] = np.ones(npts)
            A[:, 1] = time
            A[:, 2] = time2
            A[:, 3] = time3

        elif degree == 4:
            time2 = np.multiply(time, time)
            time3 = np.multiply(time2, time)
            time4 = np.multiply(time3, time)

            A[:, 0] = np.ones(npts)
            A[:, 1] = time
            A[:, 2] = time2
            A[:, 3] = time3
            A[:, 4] = time4

        elif degree == 5:
            time2 = np.multiply(time, time)
            time3 = np.multiply(time2, time)
            time4 = np.multiply(time3, time)
            time5 = np.multiply(time4, time)

            A[:, 0] = np.ones(npts)
            A[:, 1] = time
            A[:, 2] = time2
            A[:, 3] = time3
            A[:, 4] = time4
            A[:, 5] = time5

        elif degree == 6:
            time2 = np.multiply(time, time)
            time3 = np.multiply(time2, time)
            time4 = np.multiply(time3, time)
            time5 = np.multiply(time4, time)
            time6 = np.multiply(time5, time)

            A[:, 0] = np.ones(npts-1)
            A[:, 1] = time
            A[:, 2] = time2
            A[:, 3] = time3
            A[:, 4] = time4
            A[:, 5] = time5
            A[:, 6] = time6

        elif degree == 7:
            time2 = np.multiply(time, time)
            time3 = np.multiply(time2, time)
            time4 = np.multiply(time3, time)
            time5 = np.multiply(time4, time)
            time6 = np.multiply(time5, time)
            time7 = np.multiply(time6, time)

            A[:, 0] = np.ones(npts-1)
            A[:, 1] = time
            A[:, 2] = time2
            A[:, 3] = time3
            A[:, 4] = time4
            A[:, 5] = time5
            A[:, 6] = time6
            A[:, 7] = time7

        else:
            print("Illegal polynomial degree")
            # Fill in the remaining columns of the A matrix with harmonics one at a
            # time.

        col = degree + 1

        for i in range(1, nharmonics+1):
            r = np.multiply(i, np.multiply(w, time))
            A[:, col] = np.cos(r)
            col = col + 1

            A[:, col] = np.sin(r)
            col = col + 1

            # Fill in the b vector with testdata.

        b = y[:]

        # Solve for the unknown coefficients via linear least squares.

        x = np.linalg.lstsq(A, b,rcond=None)  # x = A\b

        # Assign output polynomial and Fourier coefficients.

        results = np.squeeze(x)
        coefs = results[0]

        return coefs

    # Function to compute up to 7th degree polynomial plus Fourier
    # fitted values given the pre-computed polynomial and Fourier
    # coefficients saved in a column vector. This function is
    # inted to be vectorized.
    @staticmethod
    def polyFourierCurves(coefs, w, t, degree, derivs):
        # Calculate cosine and sine harmonic arrays

        ncoefs = coefs.size  # size(coefs, 1)
        nharmonics = (ncoefs - (degree + 1)) / 2

        npts = t.size  # size(t, 1)

        # Calculate y

        a = np.zeros((npts, ncoefs))

        if degree == 0:
            a[:, 0] = 1.0

        elif degree == 1:
            a[:, 0] = 1.0
            a[:, 1] = t

        elif degree == 2:
            t2 = np.multiply(t, t)

            a[:, 0] = 1.0
            a[:, 1] = t
            a[:, 2] = t2

        elif degree == 3:
            t2 = np.multiply(t, t)
            t3 = np.multiply(t2, t)

            a[:, 0] = 1.0
            a[:, 1] = t
            a[:, 2] = t2
            a[:, 3] = t3

        elif degree == 4:
            t2 = np.multiply(t, t)
            t3 = np.multiply(t2, t)
            t4 = np.multiply(t3, t)

            a[:, 0] = 1.0
            a[:, 1] = t
            a[:, 2] = t2
            a[:, 3] = t3
            a[:, 4] = t4

        elif degree == 5:
            t2 = np.multiply(t, t)
            t3 = np.multiply(t2, t)
            t4 = np.multiply(t3, t)
            t5 = np.multiply(t4, t)

            a[:, 0] = 1.0
            a[:, 1] = t
            a[:, 2] = t2
            a[:, 3] = t3
            a[:, 4] = t4
            a[:, 5] = t5

        elif degree == 6:
            t2 = np.multiply(t, t)
            t3 = np.multiply(t2, t)
            t4 = np.multiply(t3, t)
            t5 = np.multiply(t4, t)
            t6 = np.multiply(t5, t)

            a[:, 0] = 1.0
            a[:, 1] = t
            a[:, 2] = t2
            a[:, 3] = t3
            a[:, 4] = t4
            a[:, 5] = t5
            a[:, 6] = t6

        elif degree == 7:
            t2 = np.multiply(t, t)
            t3 = np.multiply(t2, t)
            t4 = np.multiply(t3, t)
            t5 = np.multiply(t4, t)
            t6 = np.multiply(t5, t)
            t7 = np.multiply(t6, t)

            a[:, 0] = 1.0
            a[:, 1] = t
            a[:, 2] = t2
            a[:, 3] = t3
            a[:, 4] = t4
            a[:, 5] = t5
            a[:, 6] = t6
            a[:, 7] = t7

        else:
            print('Illegal polynomial degree')

        col = degree + 1

        for i in range(1, nharmonics+1):
            iw = i * w
            r = np.multiply(iw, t)
            a[:, col] = np.cos(r)
            col = col + 1

            a[:, col] = np.sin(r)
            col = col + 1

        y = a * coefs

        # Check derivatives flag

        if derivs != 1:
            fit = y
            return fit

        # Calculate yp

        ap = np.zeros((npts, ncoefs))

        if degree == 0:
            ap[:, 0] = 0.0

        elif degree == 1:
            ap[:, 0] = 0.0
            ap[:, 1] = 1

        elif degree == 2:
            ap[:, 0] = 0.0
            ap[:, 1] = 1
            ap[:, 2] = 2 * t

        elif degree == 3:
            ap[:, 0] = 0.0
            ap[:, 1] = 1
            ap[:, 2] = 2 * t
            ap[:, 3] = 3 * t2

        elif degree == 4:
            ap[:, 0] = 0.0
            ap[:, 1] = 1
            ap[:, 2] = 2 * t
            ap[:, 3] = 3 * t2
            ap[:, 4] = 4 * t3

        elif degree == 5:
            ap[:, 0] = 0.0
            ap[:, 1] = 1
            ap[:, 2] = 2 * t
            ap[:, 3] = 3 * t2
            ap[:, 4] = 4 * t3
            ap[:, 5] = 5 * t4

        elif degree == 6:
            ap[:, 0] = 0.0
            ap[:, 1] = 1
            ap[:, 2] = 2 * t
            ap[:, 3] = 3 * t2
            ap[:, 4] = 4 * t3
            ap[:, 5] = 5 * t4
            ap[:, 6] = 6 * t5

        elif degree == 7:
            ap[:, 0] = 0.0
            ap[:, 1] = 1
            ap[:, 2] = 2 * t
            ap[:, 3] = 3 * t2
            ap[:, 4] = 4 * t3
            ap[:, 5] = 5 * t4
            ap[:, 6] = 6 * t5
            ap[:, 7] = 7 * t6

        else:
            print('Illegal polynomial degree')

        col = degree + 1

        for i in range(1, nharmonics+1):
            iw = i * w

            ap[:, col] = -iw * np.sin(np.multiply(iw, t))
            col = col + 1

            ap[:, col] = iw * np.cos(np.multiply(iw, t))
            col = col + 1

        yp = ap * coefs

        # Calculate ypp

        app = np.zeros((npts, ncoefs))

        if degree == 0:
            app[:, 0] = 0.0

        elif degree == 1:
            app[:, 0] = 0.0
            app[:, 1] = 0.0

        elif degree == 2:
            app[:, 0] = 0.0
            app[:, 1] = 0.0
            app[:, 2] = 2

        elif degree == 3:
            app[:, 0] = 0.0
            app[:, 1] = 0.0
            app[:, 2] = 2
            app[:, 3] = 6 * t

        elif degree == 4:
            app[:, 0] = 0.0
            app[:, 1] = 0.0
            app[:, 2] = 2
            app[:, 3] = 6 * t
            app[:, 4] = 12 * t2

        elif degree == 5:
            app[:, 0] = 0.0
            app[:, 1] = 0.0
            app[:, 2] = 2
            app[:, 3] = 6 * t
            app[:, 4] = 12 * t2
            app[:, 5] = 20 * t3

        elif degree == 6:
            app[:, 0] = 0.0
            app[:, 1] = 0.0
            app[:, 2] = 2
            app[:, 3] = 6 * t
            app[:, 4] = 12 * t2
            app[:, 5] = 20 * t3
            app[:, 6] = 30 * t3

        elif degree == 7:
            app[:, 0] = 0.0
            app[:, 1] = 0.0
            app[:, 2] = 2
            app[:, 3] = 6 * t
            app[:, 4] = 12 * t2
            app[:, 5] = 20 * t3
            app[:, 6] = 30 * t3
            app[:, 7] = 42 * t3

        else:
            print('Illegal polynomial degree')

        col = degree + 1

        w2 = w * w

        for i in range(1, nharmonics+1):
            iw = i * w
            i2 = i * i
            i2w2 = i2 * w2

            app[:, col] = -i2w2 * np.cos(np.multiply(iw, t))
            col = col + 1

            app[:, col] = -i2w2 * np.sin(np.multiply(iw, t))
            col = col + 1

        ypp = app * coefs

        # Store results

        #fit = [y, yp, ypp]
        return Fit(y, yp, ypp)


class Fit(object):
    def __init__(self, y, yp, ypp):
        self.y = y
        self.yp = yp
        self.ypp = ypp