#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

void matmul(const float *a, const float *b, float *c, size_t m, size_t n, size_t k, bool transpose_x = false)
{
    /*
    Take two matrix and return another matrix that's their matrix multiple

    Args:
        a, b: pointers to the input matrices, a is of size m x n, b is of size n x k
        c: pointer to the output matrix, will be size m x k
        m, n, k: matrix dimensions
    */

    for (auto i = 0; i < m; i++)
    {
        for (auto j = 0; j < k; j++)
        {
            c[i * k + j] = 0.0;
            for (auto t = 0; t < n; t++)
            {
                auto a_index = transpose_x ? t * m + i : i * n + t;
                auto b_index = t * k + j;
                c[i * k + j] += a[a_index] * b[b_index];
            }
        }
    }
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
                                  float *theta, size_t m, size_t n, size_t k,
                                  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    float *z = (float *)malloc(sizeof(float) * (batch * k));
    float *grad = (float *)malloc(sizeof(float) * (n * k));
    for (auto st = 0; st < m; st += batch)
    {
        const auto *X_batch = &X[st * n];
        const auto *y_batch = &y[st];

        // z = np.exp(np.matmul(batch_X, theta))
        matmul(X_batch, theta, z, batch, n, k);
        for (auto i = 0; i < batch * k; i++)
        {
            z[i] = exp(z[i]);
        }
        // z /= np.sum(z, axis=1)[:, np.newaxis]
        for (auto i = 0; i < batch; i++)
        {
            auto rowsum = 0.0;
            for (auto j = 0; j < k; j++)
            {
                rowsum += z[i * k + j];
            }
            for (auto j = 0; j < k; j++)
            {
                z[i * k + j] /= rowsum;
            }
        }

        // e = np.zeros_like(z)
        // e[np.arange(batch_y.shape[0]), batch_y] = 1
        // z - e
        for (auto i = 0; i < batch; i++)
        {
            z[i * k + y_batch[i]] -= 1.0;
        }

        // grad = np.matmul(batch_X.T, z-e)
        matmul(X_batch, z, grad, n, batch, k, true);

        // update: theta -= (lr / batch) * grad
        const float factor = lr / batch;
        for (auto i = 0; i < n; i++)
        {
            for (auto j = 0; j < k; j++)
            {
                theta[i * k + j] -= factor * grad[i * k + j];
            }
        }
    }
    free(z);
    free(grad);
}

/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m)
{
    m.def(
        "softmax_regression_epoch_cpp",
        [](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch)
        {
            softmax_regression_epoch_cpp(
                static_cast<const float *>(X.request().ptr),
                static_cast<const unsigned char *>(y.request().ptr),
                static_cast<float *>(theta.request().ptr),
                X.request().shape[0],
                X.request().shape[1],
                theta.request().shape[1],
                lr,
                batch);
        },
        py::arg("X"), py::arg("y"), py::arg("theta"),
        py::arg("lr"), py::arg("batch"));
}
